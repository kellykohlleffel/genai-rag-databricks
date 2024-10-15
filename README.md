# :wine_glass: Create a wine country travel assistant with Fivetran and Databricks
## Scripts and code for the Fivetran + Databricks RAG-based, Gen AI Hands on Lab (60 minutes)

This repo provides the high level steps to create a RAG-based, Gen AI travel assistant using Fivetran and Databricks (detailed instructions are in the lab guide provided by your lab instructor). All required code blocks are included. This repo is the "easy button" to copy/paste each code block. If you have any issues with copy/paste, you can download the code [here](https://github.com/kellykohlleffel/genai-rag-databricks/archive/refs/heads/main.zip).

### Prerequisites and Accounts
* **Databricks Account**: mds-databricks-aws-demo
* **Unity Catalog**: ts-mds-catalog-lab
* **Fivetran Account**: MDS_DATABRICKS_HOL
* **Fivetran Destination**: DATABRICKS_LAB

> ### IMPORTANT - STEP 0: Create a Databricks Personal Compute Cluster to run a notebook
* Click in the left gray navigation
* Click Create compute
* Create a personal compute cluster that uses the following Databricks runtime 15.4 LTS ML

### STEP 1: Create a Fivetran connector to Databricks

* **Source**: Google Cloud PostgreSQL (G1 instance)
* **Fivetran Destination**: DATABRICKS_LAB
* **Schema name**: yourlastname_yourfirstname 
* **Host**: 34.94.122.157 **(see the lab guide for credentials)**
* **Schema**: agriculture
* **Table**: california_wine_country_visits

### STEP 2: View the new dataset in Databricks

* **Databricks Account**: mds-databricks-aws-demo **(see the lab guide for credentials)**
* **Unity Catalog**: ts-mds-catalog-lab
* **Schema**: yourlastname_yourfirstname_agriculture 
* **Table**: california_wine_country_visits
* Click on **Sample Data** to take a look

### STEP 3: Transform the new structured dataset into a single string to simulate an unstructured document
* Open a new query in the Databricks SQL Editor
* Select the catalog: ts-mds-catalog-lab
* Select your schema: yourlastname_yourfirstname_agriculture 
* Copy and paste this transformation script into the SQL Editor: [**transformation scripts**](01-transformations.sql)
* Click Run
* This will create a new **vineyard_data_single_string** table using the CONCAT function. Each multi-column record (winery or vineyard) will now be a single string (creates an "unstructured" document for each winery or vineyard)

```
/** Create each winery and vineyard review as a single field vs multiple fields **/
CREATE OR REPLACE TABLE vineyard_data_single_string AS
   SELECT WINERY_OR_VINEYARD, CONCAT(
       'The winery name is ', IFNULL(WINERY_OR_VINEYARD, ' Name is not known'), '.',
       ' Wine region: ', IFNULL(CA_WINE_REGION, 'unknown'),
       ' The AVA Appellation is the ', IFNULL(AVA_APPELLATION_SUB_APPELLATION, 'unknown'), '.',
       ' The website associated with the winery is ', IFNULL(WEBSITE, 'unknown'), '.',
       ' The price range is ', IFNULL(PRICE_RANGE, 'unknown'), '.',
       ' Tasting Room Hours: ', IFNULL(TASTING_ROOM_HOURS, 'unknown'), '.',
       ' The reservation requirement is: ', IFNULL(RESERVATION_REQUIRED, 'unknown'), '.',
       ' Here is a complete description of the winery or vineyard: ', IFNULL(WINERY_DESCRIPTION, 'unknown'), '.',
       ' The primary varietal this winery offers is ', IFNULL(PRIMARY_VARIETALS, 'unknown'), '.',
       ' Thoughts on the Tasting Room Experience: ', IFNULL(TASTING_ROOM_EXPERIENCE, 'unknown'), '.',
       ' Amenities: ', IFNULL(AMENITIES, 'unknown'), '.',
       ' Awards and Accolades: ', IFNULL(AWARDS_AND_ACCOLADES, 'unknown'), '.',
       ' Distance Travel Time considerations: ', IFNULL(DISTANCE_AND_TRAVEL_TIME, 'unknown'), '.',
       ' User Rating: ', IFNULL(USER_RATING, 'unknown'), '.',
       ' The secondary varietal for this winery is: ', IFNULL(SECONDARY_VARIETALS, 'unknown'), '.',
       ' Wine Styles for this winery are: ', IFNULL(WINE_STYLES, 'unknown'), '.',
       ' Events and Activities: ', IFNULL(EVENTS_AND_ACTIVITIES, 'unknown'), '.',
       ' Sustainability Practices: ', IFNULL(SUSTAINABILITY_PRACTICES, 'unknown'), '.',
       ' Social Media Channels: ', IFNULL(SOCIAL_MEDIA, 'unknown'), '.',
       ' The address is ',
           IFNULL(ADDRESS, 'unknown'), ', ',
           IFNULL(CITY, 'unknown'), ', ',
           IFNULL(STATE, 'unknown'), ', ',
           IFNULL(ZIP, 'unknown'), '.',
       ' The Phone Number is ', IFNULL(PHONE, 'unknown'), '.',
       ' Winemaker: ', IFNULL(WINEMAKER, 'unknown'),
       ' Did Kelly Kohlleffel recommend this winery?: ', IFNULL(KELLY_KOHLLEFFEL_RECOMMENDED, 'unknown')
   ) AS winery_information
   FROM california_wine_country_visits;

SELECT * FROM vineyard_data_single_string;
```

### STEP 4: Start a Databricks notebook environment to transform the vineyard_data_single_string table to a vector table and create the Gen AI application

### Step 4.1: Install the required libraries

```
# Step 4.1: Install the required libraries
%pip install transformers torch scipy 
```

### Step 4.2: Import required libraries

```
# Step 4.2: Import required libraries
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
```

### Step 4.3: Load Hugging Face Model and Tokenizer

> ## IMPORTANT: You can generate a Hugging Face token here: https://huggingface.co/settings/tokens (you'll need to create a Hugging Face account - it's free). Then update below with your Hugging Face Token.

```
# Step 4.3: Load Hugging Face Model and Tokenizer
# Update your Hugging Face token and authenticate
huggingface_token = "your_hugging_face_token"  # Replace with your Hugging Face token

# Load the tokenizer and model from Hugging Face with authentication
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", use_auth_token=huggingface_token)
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", use_auth_token=huggingface_token)
```

### Step 4.4: Define a function to generate embeddings

```
# Step 4.4: Define a function to generate embeddings
def get_embedding(text):
   inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
   outputs = model(**inputs)
   embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
   return embedding
```

### Step 4.5 Prompt for schema prefix and Unity Catalog

```
# Step 4.5: Prompt for schema prefix and Unity Catalog
schema_prefix = input("Please enter the schema prefix (e.g., 'a4_rag'): ")
unity_catalog = input("Please enter the Unity Catalog name (e.g., 'ts-mds-catalog-lab'): ")
```

### Step 4.6: Load your data and create the vector table in Unity Catalog

```
# Step 4.6: Load your data and create the vector table in Unity Catalog

# Load your vineyard data table from the new schema
df = spark.table(f"`{unity_catalog}`.`{schema_prefix}_agriculture`.`vineyard_data_single_string`")

# Convert the Spark DataFrame to a Pandas DataFrame for processing
df_pandas = df.toPandas()

# Generate embeddings for the 'winery_information' field
df_pandas['WINERY_EMBEDDING'] = df_pandas['winery_information'].apply(lambda x: get_embedding(x))

# Convert back to Spark DataFrame
df_vectors = spark.createDataFrame(df_pandas[['WINERY_OR_VINEYARD', 'winery_information', 'WINERY_EMBEDDING']])

# Save the vector table in Unity Catalog in the specified schema
df_vectors.write.format("delta").mode("overwrite").saveAsTable(f"`{unity_catalog}`.`{schema_prefix}_agriculture`.`vineyard_data_vectors`")

# Check that the table is stored
spark.sql(f"SHOW TABLES IN `{unity_catalog}`.`{schema_prefix}_agriculture`").show()
```

### Step 4.7: Define context retrieval function

```
# Step 4.7: Define context retrieval function
def get_context_from_vectors(question, top_n=3):
    # Embed the question
    question_embedding = generate_question_embedding(question)

    # Load the vectors table
    vector_query = f"""
        SELECT winery_or_vineyard, winery_information, winery_embedding
        FROM `{unity_catalog}`.`{schema_prefix}_agriculture`.`vineyard_data_vectors`
    """
    vectors_df = spark.sql(vector_query).toPandas()

    # Calculate cosine similarity for each vineyard embedding
    vectors_df["similarity"] = vectors_df["winery_embedding"].apply(
        lambda x: compute_cosine_similarity(question_embedding, x)
    )

    # Sort by similarity and retrieve the top N most similar entries
    top_context = vectors_df.sort_values(by="similarity", ascending=False).head(top_n)

    if top_context.empty:
        return None, None  # Ensure we return None for both if no context found

    # Combine the top N results into a single context and get winery names
    combined_context = " ".join(top_context["winery_information"].tolist())
    winery_names = top_context["winery_or_vineyard"].tolist()

    return combined_context, winery_names  # Ensure two values are always returned
```

### Step 4.8: Define answer generation function

```
# Step 4.8: Define answer generation function
def generate_answer(question, context):
   # For now, let's simply concatenate the context with the question as a basic example
   if not context:
       return f"Question: {question}\nAnswer: Sorry, I couldn't find relevant information based on your query."
   
   answer = f"Question: {question}\nContext: {context}\nAnswer: {context}"
   return answer

# Verify that you are referencing your new tables
# Replace 'correct_catalog_name' with the actual names you verified
spark.sql(f"SHOW TABLES IN `{unity_catalog}`.`{schema_prefix}_agriculture`").show()
```

### Step 4.9: Generate a Databricks Access Token

```
# Step 4.9: Generate a Databricks Access Token
    # Click on your profile icon in the upper right corner
    # Click on Settings
    # Click on Developer
    # Click on Access Tokens - Manage
    # Click on Generate new token, give it a unique name, and set it for 90 days
    # Add that access token to Step 4.10 line 16
```

### Step 4.10: Create an interactive wine country assistant chatbot with rich text formatting, interactive follow-up, token metrics, model selection, reset functionality, winery rating, Open Prompt & Experience-Based Prompt modes, dynamic user preferences, continuous session flow with reset, user interaction history for recommendations, and persistent engagement without exit

> ## IMPORTANT - Before running, update below with your Databricks Token

```
# Step 4.10: Create an interactive wine country assistant chatbot with rich text formatting, interactive follow-up, token metrics, model selection,
# reset functionality, winery rating, Open Prompt & Experience-Based Prompt modes, dynamic user preferences, continuous session flow
# with reset, user interaction history for recommendations, and persistent engagement without exit

from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
import time
import sys
import random

# Replace with your actual Databricks token
DATABRICKS_TOKEN = "your_databricks_token"

# Initialize the OpenAI-like Databricks client
client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://dbc-dfd1c87b-4112.cloud.databricks.com/serving-endpoints"
)

# Load the Hugging Face model and tokenizer for sentence embeddings
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# List of available models in Databricks
available_models = [
    "databricks-meta-llama-3-1-70b-instruct",
    "databricks-mixtral-8x7b-instruct",
    "databricks-dbrx-instruct",
    "databricks-meta-llama-3-1-405b-instruct",
    "databricks-llama-2-70b-chat"
]

# Dictionary to store ratings for wineries
winery_ratings = {}

# User interaction history for recommendations
user_interaction_history = []

# Simple loading animation function
def loading_animation(message):
    print(message, end="")
    for _ in range(3):
        time.sleep(0.5)
        print(".", end="")
        sys.stdout.flush()
    print()  # Move to the next line after the animation

# Function to print colored text with darker colors
def print_colored_text(text, color_code):
    # ANSI escape codes for darker text colors
    colors = {
        "31": "\033[31m",  # Dark Red
        "32": "\033[32m",  # Dark Green
        "36": "\033[36m",  # Dark Cyan (Darker Blue/Cyan)
        "bold": "\033[1m",  # Bold
        "reset": "\033[0m"  # Reset to default
    }
    # Apply the color and return formatted text
    return f"{colors.get(color_code, colors['reset'])}{text}{colors['reset']}"

# Function to format the answer as bullet points
def format_answer_in_bullets(answer):
    """
    Takes the generated answer and formats it into a list of bullet points.
    """
    # Split the answer into lines
    lines = answer.split("\n")
    
    # Add bullet points to each line and return the result
    return "\n".join([f"• {line.strip()}" for line in lines if line.strip()])

# Function to generate the embedding for the question
def generate_question_embedding(question):
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use mean pooling to get the embedding (averaging the token embeddings)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embedding

# Function to compute cosine similarity manually and map it to a 0-1 range
def compute_cosine_similarity(embedding_a, embedding_b):
    embedding_a = np.array(embedding_a).reshape(1, -1)
    embedding_b = np.array(embedding_b).reshape(1, -1)
    similarity = cosine_similarity(embedding_a, embedding_b)[0][0]

    # Ensure similarity is in the 0-1 range
    similarity = max(0, similarity)

    return similarity

# Function to count tokens used
def count_tokens(text):
    tokens = tokenizer.encode(text, return_tensors="pt")
    return tokens.shape[-1]

# Function to retrieve context from the vector embeddings based on user preferences
def get_context_from_vectors_with_preferences(wine_varietal, price_range, experience_type, specific_winery, top_n=6, similarity_threshold=0.4):
    # Step 1: Retrieve the embeddings of all wineries from the Unity Catalog vector table
    vector_query = f"""
        SELECT winery_or_vineyard, winery_information, winery_embedding
        FROM `{unity_catalog}`.`{schema_prefix}_agriculture`.`vineyard_data_vectors`
    """
    vectors_df = spark.sql(vector_query).toPandas()

    # Step 2: If a specific winery is mentioned, prioritize it
    if specific_winery:
        specific_winery_df = vectors_df[vectors_df['winery_or_vineyard'].str.contains(specific_winery, case=False, na=False)]
        if not specific_winery_df.empty:
            return specific_winery_df.iloc[0]["winery_information"], [specific_winery]

    # Step 3: Build a query string from user preferences
    query_string = f"wine varietal: {wine_varietal}, price range: {price_range}, experience type: {experience_type}"

    # Step 4: Generate the embedding for the user query
    query_embedding = generate_question_embedding(query_string)

    # Step 5: Compute cosine similarity for each winery embedding and filter by threshold
    vectors_df['similarity'] = vectors_df['winery_embedding'].apply(lambda x: compute_cosine_similarity(query_embedding, x))
    vectors_df = vectors_df[vectors_df['similarity'] >= similarity_threshold].sort_values(by='similarity', ascending=False).head(top_n)

    if vectors_df.empty:
        return None, None

    # Step 6: Return the context and winery names based on the highest similarity
    context_chunks = []
    winery_names = []
    for _, row in vectors_df.iterrows():
        context_chunks.append(row["winery_information"])
        winery_names.append(row["winery_or_vineyard"])

    return " ".join(context_chunks), winery_names

import random

# Function to get recommendations based on varietals from past interactions
def get_recommendations_based_on_varietals(winery_names, top_n=3):
    """
    This function retrieves winery recommendations based on the varietals of the last discussed wineries.
    It ensures that previously recommended wineries are not suggested again.
    """
    if not winery_names:
        return None, None

    # Get varietals of the last discussed wineries
    last_winery_varietal_query = f"""
        SELECT winery_or_vineyard, winery_information
        FROM `{unity_catalog}`.`{schema_prefix}_agriculture`.`vineyard_data_single_string`
        WHERE winery_or_vineyard IN ({', '.join([f"'{winery}'" for winery in winery_names])})
    """
    varietals_df = spark.sql(last_winery_varietal_query).toPandas()

    # Extract the varietals from the winery_information
    varietals = []
    for _, row in varietals_df.iterrows():
        info = row['winery_information']
        if "primary varietal" in info.lower():
            start = info.lower().index("primary varietal") + len("primary varietal is: ")
            end = info.lower().index(".", start)
            varietals.extend([v.strip() for v in info[start:end].split(",")])

    if not varietals:
        return None, None

    # Remove duplicates and limit to the top N varietals
    unique_varietals = list(set(varietals))

    # Fetch recommendations based on varietals
    recommendation_query = f"""
        SELECT winery_or_vineyard, winery_information
        FROM `{unity_catalog}`.`{schema_prefix}_agriculture`.`vineyard_data_single_string`
        WHERE winery_information LIKE '%{unique_varietals[0]}%' 
           OR winery_information LIKE '%{unique_varietals[1]}%' 
           OR winery_information LIKE '%{unique_varietals[2]}%'
    """
    
    recommendations_df = spark.sql(recommendation_query).toPandas()
    
    if recommendations_df.empty:
        return None, None

    # Shuffle the recommendations and avoid previously recommended wineries
    recommendations_df = recommendations_df.sample(frac=1).reset_index(drop=True)  # Shuffle the recommendations
    
    # Filter out previously recommended wineries
    recommendations_df = recommendations_df[~recommendations_df["winery_or_vineyard"].isin(user_interaction_history)]

    # Limit to top N unique recommendations
    recommendations_df = recommendations_df.head(top_n)

    # Check if we have enough unique recommendations
    if recommendations_df.empty:
        return None, None

    context_chunks = recommendations_df["winery_information"].tolist()
    winery_names = recommendations_df["winery_or_vineyard"].tolist()

    # Update the user interaction history with the newly recommended wineries
    user_interaction_history.extend(winery_names)

    return " ".join(context_chunks), winery_names

# Function to generate an answer using the selected model and calculate token metrics
def generate_answer_with_selected_model(question, context, selected_model):
    prompt = f"Question: {question}\nContext: {context}"

    start_time = time.time()

    # Start the request
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Act as a California winery visit expert for visitors to California wine country who want an incredible visit and tasting experience. You are a personal visit assistant named Fivetran+Databricks CA Wine Country Visit Assistant. Provide the most accurate information on California wineries based only on the context provided. Only provide information if there is an exact match below.  Do not go outside the context provided."},
            {"role": "user", "content": prompt}
        ],
        model=selected_model,
        max_tokens=2048,
        temperature=0.7,
        top_p=0.9
    )

    first_token_time = time.time()

    # Extract the answer
    answer = chat_completion.choices[0].message.content

    # Token counting for the answer
    total_tokens = count_tokens(answer) + count_tokens(prompt)
    time_to_first_token = first_token_time - start_time
    tokens_per_second = total_tokens / time_to_first_token if time_to_first_token > 0 else 1

    # Display token metrics
    print(f"🔢 Token Count for '{selected_model}': {total_tokens} tokens • {tokens_per_second:.2f} tokens/s • {time_to_first_token:.2f}s to first token.")

    return answer

# Function to display an interactive follow-up question or allow a new search
def ask_follow_up_question_or_new_prompt(selected_model):
    global user_interaction_history  # Ensure global history is used

    follow_up = input("Would you like more details on any of the wineries mentioned? (yes/no): ")

    if follow_up.lower() == "yes":
        winery = input("Which winery would you like more details on?: ")
        print(f"Let me gather more information on {winery} for you...\n")

        # Call the chatbot again with the specific winery request
        context, winery_names = get_context_from_vectors(winery, top_n=1)  # Fetch specific winery details
        if context:
            answer = generate_answer_with_selected_model(winery, context, selected_model)
            print(f"\nWinery names found: {', '.join(winery_names)}\n")
            print(f"Context retrieved from vineyard data:\n{context}\n")
            print(f"🍷 **Answer**:\n{answer}")

            # Add to interaction history
            user_interaction_history.extend([winery for winery in winery_names if winery not in user_interaction_history])

            # Continue asking for follow-up questions or new prompt after displaying details
            return ask_follow_up_question_or_new_prompt(selected_model)
        else:
            print("Sorry, I couldn't find specific information for that winery.")
            return ask_follow_up_question_or_new_prompt(selected_model)

    elif follow_up.lower() == "no":
        # Now prompt the user whether they want to reset or continue with a new prompt
        next_step = input("Would you like to create a new prompt or reset the model selection? (new prompt/reset): ")
        if next_step.lower() == "new prompt":
            return True  # This will allow the chatbot to continue with a new prompt
        elif next_step.lower() == "reset":
            print(print_colored_text("\n🔄 Resetting to model selection...\n", "36"))
            run_chatbot()  # Reset to the start (model selection) without ending the session
        else:
            print("Please respond with 'new prompt' or 'reset'.")
            return ask_follow_up_question_or_new_prompt(selected_model)

    else:
        print("Please respond with 'yes' or 'no'.")
        return ask_follow_up_question_or_new_prompt(selected_model)

# Function to provide a summary of the wineries discussed
def provide_summary(winery_names):
    summary = f"Here's a quick recap of the wineries discussed: {', '.join(winery_names)}."
    print(f"🍷 **Summary**:\n{summary}")

# Function to select a model
def select_model():
    print("🧑‍💻 Pick a model to use for your wine country travel assistant:")
    for i, model_name in enumerate(available_models, 1):
        print(f"{i}. {model_name}")
  
    while True:
        try:
            selected_number = int(input("Select a model (enter a number): "))
            if 1 <= selected_number <= len(available_models):
                return available_models[selected_number - 1]
            else:
                print("Invalid selection. Please enter a number corresponding to a model.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Function to select between Open Prompt and Experience-Based Prompt
def select_prompt_mode():
    print("Choose the type of prompt you'd like to use:")
    print("1. Open Prompt")
    print("2. Experience-Based Prompt")
    if user_interaction_history:
        print("3. Surprise me with some recommendations!")

    while True:
        try:
            selected_mode = int(input("Select a prompt mode (enter a number): "))
            if selected_mode == 1:
                return "open"
            elif selected_mode == 2:
                return "experience"
            elif selected_mode == 3 and user_interaction_history:
                return "recommendations"
            else:
                print("Invalid selection. Please enter 1 for Open Prompt, 2 for Experience-Based Prompt, or 3 for recommendations.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Function to allow users to rate wineries in real-time
def rate_winery(winery_names):
    print("\nWould you like to rate any of the wineries? (yes/no)")
    if input().lower() == "yes":
        for winery in winery_names:
            while True:  # Loop until a valid rating is provided
                try:
                    rating = int(input(f"Please rate {winery} (1-5 stars): "))
                    if 1 <= rating <= 5:
                        winery_ratings[winery] = rating
                        print(f"Thank you! You rated {winery} {rating} stars.")
                        break  # Exit the loop once a valid rating is entered
                    else:
                        print("Invalid rating. Please enter a number between 1 and 5.")
                except ValueError:
                    print("Invalid input. Please enter a valid number.")

    print("\nCurrent Ratings:")
    for winery, rating in winery_ratings.items():
        print(f"{winery}: {rating} stars")

# Function to provide third-party review links (Yelp/TripAdvisor)
def provide_review_links(winery_names):
    print("\nFor additional reviews, you can visit the following sites:")
    for winery in winery_names:
        print(f"{winery}: [Yelp](https://www.yelp.com/search?find_desc={winery.replace(' ', '%20')}) | [TripAdvisor](https://www.tripadvisor.com/Search?q={winery.replace(' ', '%20')})")

# Main chatbot function with continuous query and exit logic
def run_chatbot():
    selected_model = select_model()  # Select the model first
    print(f"Using model: {selected_model}")
  
    continue_chat = True  # Flag to control the loop

    while continue_chat:
        search_mode = select_prompt_mode()  # Prompt for Open, Experience-Based, or Recommendations Prompt

        if search_mode == "experience":
            # Trigger the experience-based search flow
            print("You're starting the Experience-Based Prompt!")
          
            # Collect user preferences for the experience-based search
            wine_varietal = select_wine_varietal()
            price_range = select_price_range()
            experience_type = select_experience_type()
            specific_winery = input("Do you have a specific winery you want to visit? (optional): ")
          
            # Display "Retrieving information" message
            loading_animation("Retrieving information")  # Simulate loading

            # Retrieve context based on user preferences
            context, winery_names = get_context_from_vectors_with_preferences(
                wine_varietal, price_range, experience_type, specific_winery, top_n=3
            )

            if not context:
                print(print_colored_text("Sorry, I couldn't find specific information for those preferences.", "31"))  # Red for errors
            else:
                # Generate an answer using the selected model and token metrics
                answer = generate_answer_with_selected_model(f"Preferences: {wine_varietal}, {price_range}, {experience_type}", context, selected_model)

                # Display the winery names and the generated answer
                print(print_colored_text(f"\nWinery names found: {', '.join(winery_names)}\n", "32"))  # Green for success
                print(print_colored_text("🍷 **Answer**:\n", "bold"))  # Bold and with a wine glass icon
                print(format_answer_in_bullets(answer))  # Format answer as bullet points

                provide_summary(winery_names)  # Provide a summary of the wineries mentioned
                rate_winery(winery_names)  # Allow the user to rate wineries
                provide_review_links(winery_names)  # Provide third-party review links (Yelp, TripAdvisor)

                # Add winery names to the interaction history
                user_interaction_history.extend([winery for winery in winery_names if winery not in user_interaction_history])
          
        elif search_mode == "recommendations":
            # Trigger recommendations flow
            print("Based on your previous interactions, I have some recommendations for you!")
            loading_animation("Retrieving recommendations")  # Simulate loading

            # Retrieve recommendations based on user history
            context, winery_names = get_recommendations_based_on_varietals(user_interaction_history, top_n=3)

            if not context:
                print(print_colored_text("Sorry, I couldn't find recommendations based on your history.", "31"))  # Red for errors
            else:
                # Generate an answer using the selected model and token metrics
                answer = generate_answer_with_selected_model("Recommendations based on your past interactions", context, selected_model)

                # Display the winery names and the generated answer
                print(print_colored_text(f"\nRecommended Winery names: {', '.join(winery_names)}\n", "32"))  # Green for success
                print(print_colored_text("🍷 **Recommendations**:\n", "bold"))  # Bold and with a wine glass icon
                print(format_answer_in_bullets(answer))  # Format answer as bullet points

                provide_summary(winery_names)  # Provide a summary of the wineries mentioned
                rate_winery(winery_names)  # Allow the user to rate wineries
                provide_review_links(winery_names)  # Provide third-party review links (Yelp, TripAdvisor)

                # Add winery names to the interaction history
                user_interaction_history.extend([winery for winery in winery_names if winery not in user_interaction_history])

        else:
            # Trigger the open prompt flow
            print("You're starting the Open Prompt!")
            question = input("🍇 : Ask me anything about California wine country or type 'reset' to start over: ")
          
            if question.lower() == 'reset':
                print(print_colored_text("\n🔄 Resetting the conversation...\n", "36"))
                run_chatbot()  # Reset the conversation and prompt model selection
                break  # Exit the current loop and start over
          
            loading_animation("Retrieving information")  # Simulate loading
          
            # Retrieve the relevant context based on the question
            context, winery_names = get_context_from_vectors(question, top_n=3)  # Adjust top_n for multiple winery retrieval
          
            if not context:
                print(print_colored_text("Sorry, I couldn't find specific information for that prompt.", "31"))  # Red for errors
            else:
                # Generate an answer using the selected model and token metrics
                answer = generate_answer_with_selected_model(question, context, selected_model)

                # Display the winery names and the generated answer
                print(print_colored_text(f"\nWinery names found: {', '.join(winery_names)}\n", "32"))  # Green for success
                print(print_colored_text("🍷 **Answer**:\n", "bold"))  # Bold and with a wine glass icon
                print(format_answer_in_bullets(answer))  # Format answer as bullet points

                provide_summary(winery_names)  # Provide a summary of the wineries mentioned
                rate_winery(winery_names)  # Allow the user to rate wineries
                provide_review_links(winery_names)  # Provide third-party review links (Yelp, TripAdvisor)

                # Add winery names to the interaction history
                user_interaction_history.extend([winery for winery in winery_names if winery not in user_interaction_history])
      
        # After each query or experience-based result, ask if the user wants to continue
        continue_chat = ask_follow_up_question_or_new_prompt(selected_model)

    print("Chat session ended.")

# Function to simulate a dropdown selection for wine varietal
def select_wine_varietal():
    wine_varietals = ['Pinot Noir', 'Chardonnay', 'Cabernet Sauvignon', 'Sauvignon Blanc', 'Zinfandel']
    print("Please choose a wine varietal:")
    for i, varietal in enumerate(wine_varietals, 1):
        print(f"{i}. {varietal}")
    while True:
        try:
            selected_varietal = int(input("Select a wine varietal (enter a number): "))
            if 1 <= selected_varietal <= len(wine_varietals):
                return wine_varietals[selected_varietal - 1]
            else:
                print("Invalid selection. Please enter a number corresponding to a wine varietal.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Function to simulate a dropdown selection for price range
def select_price_range():
    price_ranges = ['$', '$$', '$$$', '$$$$']
    print("Please choose a price range:")
    for i, price in enumerate(price_ranges, 1):
        print(f"{i}. {price}")
    while True:
        try:
            selected_price = int(input("Select a price range (enter a number): "))
            if 1 <= selected_price <= len(price_ranges):
                return price_ranges[selected_price - 1]
            else:
                print("Invalid selection. Please enter a number corresponding to a price range.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Function to simulate a dropdown selection for experience type
def select_experience_type():
    experience_types = ['Boutique', 'Grand Estate', 'Family-owned', 'Organic', 'Vineyard Tours']
    print("Please choose an experience type:")
    for i, experience in enumerate(experience_types, 1):
        print(f"{i}. {experience}")
    while True:
        try:
            selected_experience = int(input("Select an experience type (enter a number): "))
            if 1 <= selected_experience <= len(experience_types):
                return experience_types[selected_experience - 1]
            else:
                print("Invalid selection. Please enter a number corresponding to an experience type.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Run the chatbot
run_chatbot()
```

### Step 5: Have some fun checking out the travel assistant features and creating prompts for unique visits using RAG
* Test the new application with your own prompts or check out the sample prompts in the lab guide
* Control records in the postgresql database (for testing RAG) include: kohlleffel vineyards, millman estate, hrncir family cellars, tony kelly pamont vineyards, and kai lee family cellars

### Fivetran + Databricks California Wine Country Visit Assistant

![Wine Country Visit Assistant Notebook Screenshot](images/2024-10-10%20Databricks%20Notebook%20-%20Travel%20Assistant.png)

-----
