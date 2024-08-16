from transformers import pipeline

# Load a pre-trained text-generation model
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")


def chat():
    print("Chatbot: Hello! How can I help you today?")
    chat_history = ""

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break

        # Append user input to chat history
        chat_history += f"User: {user_input}\n"

        # Generate a response
        response = chatbot(chat_history, max_length=1000, pad_token_id=50256)[0]['generated_text']

        # Extract the chatbot's response from the generated text
        chatbot_response = response.split('\n')[-1].replace("Chatbot: ", "")

        # Print the response
        print(f"Chatbot: {chatbot_response}")

        # Update chat history with chatbot's response
        chat_history += f"Chatbot: {chatbot_response}\n"


if __name__ == "__main__":
    chat()
