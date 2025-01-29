from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define the system prompt
system_prompt = "You are a helpful assistant that provides concise and accurate answers."

# Create the ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Simulated chat history and user input
chat_history = [
    {"role": "human", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]

user_input = "What is the population of France?"

# Use the prompt template to format the messages
formatted_prompt = chat_prompt.format_messages(
    chat_history=chat_history,
    input=user_input
)

# The `formatted_prompt` now contains a structured format that can be passed to an AI model.
print(formatted_prompt)