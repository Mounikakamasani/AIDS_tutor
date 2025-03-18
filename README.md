# AI Conversational Data Science Tutor

This project is a Conversational AI Tutor specialized in Data Science, built using **Gemini 1.5 Pro**, **LangChain**, and **Streamlit**. It helps users resolve data science queries while maintaining conversation awareness using memory.

## Features
- 🧠 **Conversational Memory**: Remembers previous interactions for better context.
- 📊 **Data Science Focused**: Answers only data science-related queries.
- ⚡ **Powered by Gemini 1.5 Pro**: Uses Google's latest AI model for responses.
- 💬 **Interactive Chat UI**: Built using Streamlit for a seamless user experience.

## Installation
### Prerequisites
Ensure you have Python installed (>=3.8) and set up your Google API key.

### Setup Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ai-data-science-tutor.git
   cd ai-data-science-tutor
   ```
2. Install dependencies:
   ```bash
   pip install streamlit langchain google-generativeai
   ```
3. Set up your Google API key:
   ```bash
   export GOOGLE_API_KEY="your_google_api_key"
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
- Open the chat interface in your browser.
- Ask data science-related questions.
- The AI will respond while maintaining conversation history.

## Deployment
To deploy on **Streamlit Cloud**, commit your changes and push to GitHub, then link the repository in Streamlit Cloud.

## Future Enhancements
- 🔄 **Streaming Responses** for real-time AI interaction.
- 🌎 **Deployment on Cloud Platforms** (AWS/GCP).
- 🛠 **Fine-tuning AI Responses** for improved accuracy.

## Contributing
Pull requests are welcome! Feel free to contribute to improve functionality.

## License
This project is licensed under the MIT License.

