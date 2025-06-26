from pymongo import MongoClient
from typing import Dict, List
from openai import OpenAI
import json
from datetime import datetime
import re
from bson import json_util
import os



class LLMQueryGenerator:
    """Uses an LLM to generate queries and interpret results"""

    def __init__(self, llm_api_key: str, db_schema: Dict):
        """
        Initialize the LLM query generator

        Args:
            llm_api_key: API key for the LLM service
            db_schema: Dictionary describing the database schema
        """
        self.client = OpenAI(api_key=llm_api_key)
        self.db_schema = db_schema

    def generate_query(self, question: str, collection: str) -> Dict:
        """Generate a database query from a natural language question"""
        prompt = f"""
        You are a MongoDB query generator. Based on the user's question, generate an appropriate MongoDB query.
        The database schema is:
        {json.dumps(self.db_schema, indent=2)}

        For collection '{collection}', generate either:
        1. A find query with filter and optional projection
        2. An aggregation pipeline

        Return ONLY a JSON object with either:
        - 'filter' and optional 'projection' for find queries
        - 'pipeline' for aggregation queries
        - Always include 'collection' specifying which collection to query

        User question: {question}
        """

        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.3
        )

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # Fallback to a simple query if LLM response is invalid
            return {
                "collection": collection,
                "filter": {},
                "projection": {"_id": 0},
                "limit": 5
            }

    def interpret_results(self, results: List[Dict], question: str) -> str:
        """Convert database results into a natural language response"""
        if not results:
            return "I couldn't find any data matching your query."

        results_str = json_util.dumps(results, indent=2)[:3000]  # Truncate to avoid token limits

        prompt = f"""
        You are a data analyst assistant. Based on the following database results, 
        provide a clear and concise answer to the user's question.

        User question: {question}

        Database results (JSON format):
        {results_str}

        Provide your response in plain text, focusing on answering the question directly.
        Include relevant numbers and facts from the data.
        """

        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5
        )

        return response.choices[0].message.content


class MultiStepQueryProcessor:
    """Handles multi-step queries using LLM guidance"""

    def __init__(self, llm_generator: LLMQueryGenerator, db):
        self.llm = llm_generator
        self.db = db

    def process_question(self, question: str) -> str:
        """Process a complex, multi-step question"""
        if self._is_product_comparison(question):
            return self._handle_product_comparison(question)
        # First have the LLM plan the steps
        planning_prompt = f"""
        You are a data analysis assistant working with MongoDB. The user has asked:
        "{question}"

        The available collections are:
        {json.dumps(list(self.llm.db_schema.keys()), indent=2)}

        If this question requires querying multiple collections or performing multiple steps,
        outline the steps needed to answer it. For each step, specify:
        1. Which collection to query
        2. What information to get
        3. How it contributes to answering the overall question

        Return your response as a JSON array of steps, where each step has:
        - "purpose": Why this step is needed
        - "collection": Which collection to query
        - "query": Either a find filter or aggregation pipeline
        """

        planning_response = self.llm.client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "system", "content": planning_prompt}],
            temperature=0.3
        )

        try:
            steps = json.loads(planning_response.choices[0].message.content)
            if not isinstance(steps, list):
                steps = [steps]
        except:
            return "I couldn't formulate a plan to answer this complex question."

        # Execute each step and collect results
        step_results = []
        for step in steps:
            try:
                if "pipeline" in step["query"]:
                    results = list(self.db[step["collection"]].aggregate(step["query"]["pipeline"]))
                else:
                    results = list(self.db[step["collection"]].find(
                        step["query"].get("filter", {}),
                        step["query"].get("projection", {})
                    ))
                step_results.append({
                    "purpose": step["purpose"],
                    "results": json_util.dumps(results)
                })
            except Exception as e:
                step_results.append({
                    "purpose": step["purpose"],
                    "error": str(e)
                })

        # Have the LLM synthesize the final answer
        synthesis_prompt = f"""
        Original question: {question}

        The following steps were executed to gather information:
        {json.dumps(step_results, indent=2)}

        Please provide a comprehensive answer to the original question based on these results.
        If any steps failed, acknowledge that some data might be incomplete.
        """

        synthesis_response = self.llm.client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "system", "content": synthesis_prompt}],
            temperature=0.5
        )

        return synthesis_response.choices[0].message.content

    def _handle_product_comparison(self, question: str) -> str:
        """Specialized handler for product comparison queries with better categorization"""
        try:
            # Step 1: Categorize products and get account stats
            product_stats = list(self.db.accounts.aggregate([
                {"$unwind": "$products"},
                {"$group": {
                    "_id": {
                        "$switch": {
                            "branches": [
                                {"case": {"$regexMatch": {"input": "$products", "regex": "credit|investment|stock",
                                                          "options": "i"}},
                                 "then": "credit"},
                                {"case": {"$regexMatch": {"input": "$products", "regex": "loan|mortgage|debt",
                                                          "options": "i"}},
                                 "then": "loan"},
                            ],
                            "default": "other"
                        }
                    },
                    "account_count": {"$sum": 1},
                    "avg_limit": {"$avg": "$limit"},
                    "account_ids": {"$push": "$account_id"}
                }}
            ]))

            if not product_stats:
                return "No product distribution data found."

            # Step 2: Get transaction stats for each category
            comparisons = []
            for category in product_stats:
                account_ids = category["account_ids"]
                tx_stats = list(self.db.transactions.aggregate([
                    {"$match": {"account_id": {"$in": account_ids}}},
                    {"$group": {
                        "_id": None,
                        "avg_transactions": {"$avg": "$transaction_count"},
                        "total_transactions": {"$sum": "$transaction_count"},
                        "active_accounts": {
                            "$sum": {
                                "$cond": [{"$gt": ["$transaction_count", 0]}, 1, 0]
                            }
                        }
                    }}
                ]))

                comparisons.append({
                    "category": category["_id"],
                    "account_count": category["account_count"],
                    "avg_limit": category["avg_limit"],
                    "tx_stats": tx_stats[0] if tx_stats else {}
                })

            # Step 3: Format for LLM with explicit categories
            results = {
                "question": question,
                "available_categories": [c["category"] for c in comparisons],
                "comparisons": comparisons,
                "timestamp": datetime.now().isoformat()
            }

            # Step 4: Special handling when expected categories are missing
            expected_categories = {"credit", "loan"}
            available_categories = set(results["available_categories"])
            missing_categories = expected_categories - available_categories

            if missing_categories:
                results["analysis_notes"] = {
                    "missing_categories": list(missing_categories),
                    "possible_reasons": [
                        "Data not available in database",
                        "Different product naming conventions",
                        "No accounts of this type currently exist"
                    ]
                }

            return self.llm.interpret_results([results], question)

        except Exception as e:
            return f"Error processing product comparison: {str(e)}"

    def _is_product_comparison(self, question: str) -> bool:
        q = question.lower()
        return ("compare" in q) and ("product" in q or "products" in q)


class Chatbot:
    """MongoDB Analytics Chatbot that uses LLM for query generation and response"""

    def __init__(self):
        """
        Initialize the chatbot

        Args:
            connection_string: MongoDB Atlas connection string
            llm_api_key: API key for the LLM service
            db_name: Database name (defaults to 'sample_analytics')
        """
        connection_string = os.environ['db_connection_string']
        llm_api_key = os.environ['OPENAI_API_KEY']
        db_name = "sample_analytics"
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]

        # Define the database schema for the LLM
        self.db_schema = {
            "customers": {
                "description": "Customer information",
                "fields": {
                    "_id": "ObjectId",
                    "username": "string",
                    "name": "string",
                    "address": "string",
                    "birthdate": "ISODate",
                    "email": "string",
                    "accounts": "array of account_ids"
                }
            },
            "accounts": {
                "description": "Bank account information",
                "fields": {
                    "_id": "ObjectId",
                    "account_id": "integer",
                    "limit": "integer",
                    "products": "array of strings"
                }
            },
            "transactions": {
                "description": "Financial transactions",
                "fields": {
                    "_id": "ObjectId",
                    "account_id": "integer",
                    "transaction_count": "integer",
                    "bucket_start_date": "ISODate",
                    "bucket_end_date": "ISODate",
                    "transactions": "array of transaction details"
                }
            }
        }

        # Initialize the LLM components
        self.llm_generator = LLMQueryGenerator(llm_api_key, self.db_schema)
        self.multistep_processor = MultiStepQueryProcessor(self.llm_generator, self.db)

    def ask(self, question: str) -> str:
        """
        Answer a user's question by generating and executing appropriate queries

        Args:
            question: The user's natural language question

        Returns:
            A natural language response based on the query results
        """
        try:
            # Handle all location questions consistently
            if self._is_location_question(question):
                query = self._generate_location_query(question)

                if not query:
                    return "Please specify a valid location in your question"

                results = list(self.db[query["collection"]].find(
                    query.get("filter", {}),
                    query.get("projection", {})
                ))

                if not results:
                    regex_pattern = query['filter']['$or'][0]['address']['$regex']
                    location = regex_pattern.split('\\b')[1] if '\\b' in regex_pattern else "the specified location"
                    return f"No customers found in {location}"

                return self._format_location_results(results, question)

        except Exception as e:
            return f"Error processing location query: {str(e)}"
        try:
            transaction_keywords = [
                "transactions by", "transactions made by", "transactions did",
                "purchases by", "activity of", "history of"
            ]
            if any(kw in question.lower() for kw in transaction_keywords):
                return self._handle_user_transaction_query(question)
            # First determine if this is a multi-step question
            if "compare" in question.lower() and ("product" in question.lower() or "products" in question.lower()):
                query = self._generate_product_comparison_query(question)
                results = list(self.db[query["collection"]].aggregate(query["pipeline"]))
                return self.llm_generator.interpret_results(results, question)

            if self._is_complex_question(question):
                return self.multistep_processor.process_question(question)

            # Determine which collection is most relevant
            collection = self._determine_collection(question)

            # Generate the query using LLM - with special handling for date queries
            if "born after" in question.lower() or "birthdate" in question.lower():
                # Special handling for date queries
                query = self._generate_date_query(question, collection)
            else:
                query = self.llm_generator.generate_query(question, collection)

            # Execute the query
            if "pipeline" in query:
                results = list(self.db[query["collection"]].aggregate(query["pipeline"]))
            else:
                results = list(self.db[query["collection"]].find(
                    query.get("filter", {}),
                    query.get("projection", {})
                ))

            # Have the LLM interpret the results
            return self.llm_generator.interpret_results(results, question)

        except Exception as e:
            return f"An error occurred while processing your request: {str(e)}"

    def _is_complex_question(self, question: str) -> bool:
        """Determine if a question requires multi-step processing"""
        complexity_keywords = [
            "compare", "relationship", "across", "between",
            "versus", "vs", "correlate", "combination"
        ]
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in complexity_keywords)

    def _determine_collection(self, question: str) -> str:
        """Determine which collection is most relevant to the question"""
        # Let the LLM help with this determination
        prompt = f"""
        Based on the following question, determine which MongoDB collection is most relevant:
        "{question}"

        Available collections:
        - customers: Customer personal information
        - accounts: Bank account details
        - transactions: Financial transaction records

        Return ONLY the collection name (customers, accounts, or transactions).
        """

        response = self.llm_generator.client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.1
        )

        collection = response.choices[0].message.content.lower()
        return collection if collection in ["customers", "accounts", "transactions"] else "customers"

    def _handle_user_transaction_query(self, question: str) -> str:
        """Handle transaction queries by username, email, or name"""
        try:
            # Extract user identifier from question
            prompt = f"""
            Extract the user identifier from this question. It could be:
            - Full name (e.g., "Elizabeth Ray")
            - Email address (e.g., "user@example.com")
            - Username (e.g., "ejray")

            Return ONLY the identifier. If none found, return "None".
            Question: "{question}"
            """

            response = self.llm_generator.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )
            user_identifier = response.choices[0].message.content.strip()

            if user_identifier.lower() == "none":
                return "Could not determine the user identifier from your question."

            # Step 1: Find the user's account IDs using all possible identifiers
            customer = self.db.customers.find_one({
                "$or": [
                    {"name": {"$regex": f"^{user_identifier}$", "$options": "i"}},
                    {"email": {"$regex": f"^{user_identifier}$", "$options": "i"}},
                    {"username": {"$regex": f"^{user_identifier}$", "$options": "i"}}
                ]
            }, {"_id": 0, "accounts": 1, "name": 1})

            if not customer or not customer.get("accounts"):
                return f"No accounts found for: {user_identifier}"

            account_ids = customer["accounts"]
            user_name = customer.get("name", user_identifier)

            # Step 2: Get detailed transaction stats
            transaction_stats = list(self.db.transactions.aggregate([
                {"$match": {"account_id": {"$in": account_ids}}},
                {"$group": {
                    "_id": None,
                    "total_transactions": {"$sum": "$transaction_count"},
                    "total_amount": {"$sum": "$amount"},
                    "avg_amount": {"$avg": "$amount"},
                    "first_transaction": {"$min": "$date"},
                    "last_transaction": {"$max": "$date"}
                }}
            ]))

            if not transaction_stats:
                return f"No transactions found for {user_name}"

            stats = transaction_stats[0]
            return (
                f"Transaction summary for {user_name}:\n"
                f"• Total transactions: {stats['total_transactions']}\n"
                f"• Total amount: ${stats['total_amount']:,.2f}\n"
                f"• Average transaction: ${stats['avg_amount']:,.2f}\n"
                f"• First transaction: {stats['first_transaction'].strftime('%Y-%m-%d')}\n"
                f"• Last transaction: {stats['last_transaction'].strftime('%Y-%m-%d')}"
            )

        except Exception as e:
            return f"Error processing transaction query: {str(e)}"



    def _generate_date_query(self, question: str, collection: str) -> Dict:
        """Special handling for date-based queries with improved date extraction"""
        # Extract year using regular expression
        year_match = re.search(r'(19|20)\d{2}', question)
        year = int(year_match.group(0)) if year_match else None

        # Handle "born after X" or "born before X"
        comparison = None
        if "after" in question.lower():
            comparison = "$gt"
        elif "before" in question.lower():
            comparison = "$lt"
        elif "in" in question.lower() and year:
            # For "born in 1980" queries
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
            return {
                "collection": "customers",
                "filter": {
                    "birthdate": {"$gte": start_date, "$lte": end_date}
                },
                "projection": {"_id": 0, "name": 1, "birthdate": 1}
            }

        if not year or not comparison:
            return {
                "collection": collection,
                "filter": {},
                "projection": {"_id": 0, "name": 1, "birthdate": 1},
                "limit": 5
            }

        # Create proper MongoDB date query
        start_date = datetime(year, 1, 1)
        return {
            "collection": "customers",
            "filter": {
                "birthdate": {comparison: start_date}
            },
            "projection": {"_id": 0, "name": 1, "birthdate": 1}
        }

    def _generate_product_comparison_query(self, question: str) -> Dict:
        """Special handling for product comparison queries"""
        return {
            "collection": "accounts",
            "pipeline": [
                {"$match": {"products": {"$exists": True}}},
                {"$unwind": "$products"},
                {"$group": {
                    "_id": "$products",
                    "count": {"$sum": 1},
                    "avg_limit": {"$avg": "$limit"},
                    "account_ids": {"$push": "$account_id"}
                }},
                {"$sort": {"count": -1}}
            ]
        }

    def _extract_location_from_question(self, question: str):
        """Use NLP to extract location from question"""
        prompt = f"""
        Extract just the location name from this question. Return ONLY the location name.
        Question: "{question}"

        Examples:
        - "people in Boston" → "Boston"
        - "customers from New York" → "New York" 
        - "who lives in San Francisco" → "San Francisco"
        """

        try:
            response = self.llm_generator.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1,
                max_tokens=30
            )
            location = response.choices[0].message.content.strip()
            return location if location.lower() != "none" else None
        except:
            return None

    def _generate_location_query(self, question: str) -> Dict:
        """Generate consistent location queries for all location questions"""
        # First extract location using both LLM and regex
        location = self._extract_location_from_question(question)

        if not location:
            # Enhanced regex pattern to catch more location query formats
            patterns = [
                r'(?:live in|living in|from|residing in|address|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'in\s+(\w+)$'  # Catches "people in Boston"
            ]
            for pattern in patterns:
                match = re.search(pattern, question, re.I)
                if match:
                    location = match.group(1)
                    break

        if not location:
            return None

        return {
            "collection": "customers",
            "filter": {
                "$or": [
                    {"address": {"$regex": f"\\b{re.escape(location)}\\b", "$options": "i"}},
                    {"address.city": {"$regex": f"^{re.escape(location)}$", "$options": "i"}}
                ]
            },
            "projection": {"_id": 0, "name": 1, "email": 1, "address": 1}
        }

    def _format_location_results(self, results: List[Dict], question: str) -> str:
        """Consistent formatting for all location results"""
        location = self._extract_location_from_question(question) or "the specified location"

        if "how many" in question.lower():
            return f"There are {len(results)} customers living in {location}."

        response = [f"Customers living in {location}:"]
        for i, customer in enumerate(results, 1):
            name = customer.get('name', 'Unknown')
            email = customer.get('email', 'No email')
            address = self._format_address(customer.get('address', {}))
            response.append(f"{i}. {name} - {email}\n   Address: {address}")

        return "\n\n".join(response)

    def _format_address(self, address) -> str:
        """Format address consistently whether string or object"""
        if isinstance(address, str):
            return address
        if isinstance(address, dict):
            return ", ".join(filter(None, [
                address.get('street'),
                address.get('city'),
                address.get('state'),
                address.get('zip')
            ]))
        return str(address)

    def _is_location_question(self, question: str) -> bool:
        """Determine if this is a location-based question"""
        location_keywords = ["living in", "from", "residing in", "address", "in", "live in"]
        question_lower = question.lower()
        return any(kw in question_lower for kw in location_keywords)
