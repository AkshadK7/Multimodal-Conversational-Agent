# Multimodal Conversational Agent

A chat-based agent utilizing the Mistral 7B Large Language Model (LLM), Langchain, Ollama, and Streamlit to answer questions about files through Retrieval-Augmented Generation (RAG).

## Overview

The Multimodal Conversational Agent is a chat-based system designed to interpret user queries and retrieve relevant information from specified files using a large language model (LLM). By leveraging Retrieval-Augmented Generation (RAG), the agent enhances its responses by retrieving pertinent data from external knowledge bases, enabling it to generate more informative and contextually accurate answers. This approach allows the agent to provide comprehensive responses to a wider range of questions while maintaining relevance to the given document.


## Repository Contents

- `dump/`: Directory containing sample files for the agent to process.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `README.md`: Project documentation.
- `app.py`: Main application script integrating the LLM with Streamlit for the user interface.
- `requirements.txt`: List of Python dependencies required for the project.

## Requirements

- Python 3.x
- Streamlit
- Langchain
- Ollama

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AkshadK7/Multimodal-Conversational-Agent.git
   cd Multimodal-Conversational-Agent
   ```

2. **Install Dependencies**:
   It's recommended to use a virtual environment to manage dependencies.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage

- **Application Interface**:
  - Upon running, the Streamlit application will provide an interface where users can input queries.
  - The agent processes these queries, searches within the provided files, and returns relevant information based on the content and external knowledge bases.

## Running Mistral 7B Locally using Ollama 🦙

Ollama allows you to run open-source large language models, such as Llama 2, locally. It bundles model weights, configuration, and data into a single package, defined by a Modelfile, optimizing setup and configuration details, including GPU usage.

**For Mac and Linux Users:**
Ollama effortlessly integrates with Mac and Linux systems, offering a user-friendly installation process. Mac and Linux users can swiftly set up Ollama to access its rich features for local language model usage. Detailed instructions can be found here: [Ollama GitHub Repository for Mac and Linux](https://github.com/ollama/ollama).

**For Windows Users:**
For Windows users, the process involves a few additional steps, ensuring a smooth Ollama experience:

1. **Install WSL 2:** To enable WSL 2, kindly refer to the official Microsoft documentation for comprehensive installation instructions: [Install WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install).

2. **Install Docker:** Docker for Windows is a crucial component. Installation guidance is provided in the official Docker documentation: [Install Docker for Windows](https://docs.docker.com/desktop/install/windows-install).

3. **Utilize Docker Image:** Windows users can access Ollama by using the Docker image provided here: [Ollama Docker Image](https://hub.docker.com/r/ollama/ollama).

Now you can easily use Mistral in the command line (CMD) using the following command:

```
docker exec -it ollama ollama run mistral
```

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/AkshadK7/Multimodal-Conversational-Agent/blob/master/LICENSE) file for details.

## Acknowledgements

Special thanks to the developers of Mistral 7B LLM, Langchain, Ollama, and the Streamlit community for their invaluable tools and support.
```

*Note: Ensure that the `requirements.txt` file includes all necessary dependencies for the project. If it doesn't exist, you may need to create it by listing the required packages.* 
















