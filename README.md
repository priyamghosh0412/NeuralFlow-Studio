# NeuralFlow Studio

NeuralFlow Studio is a comprehensive, AI-powered Data Science and Engineering platform designed to streamline the end-to-end data lifecycle. It integrates Exploratory Data Analysis (EDA), Intelligent Data Transformation, Visualization, and Agentic Workflows into a unified interface.

## 🚀 Key Features

### 1. Exploratory Data Analysis (EDA)
-   **Data Connection**: Seamlessly connect to PostgreSQL databases or load CSV datasets.
-   **Deep Profiling**: Automatically generate detailed statistics for numerical and categorical columns (missing values, distributions, unique counts).
-   **SQL Interface**: Built-in SQL editor to query connected databases directly.
-   **Visual Insights**: Automatic generation of data quality insights and alerts.

### 2. Intelligent Transformation (AI-Powered)
-   **Natural Language Processing**: Use the built-in AI Assistant (powered by **Ollama**) to transform data using plain English (e.g., "Extract week number from date", "Remove outliers").
-   **Transformation Catalog**: Access a library of pre-built standard transformations (Scaling, Encoding, Imputation, etc.).
-   **Interactive Pipeline**: Build a sequence of transformations with drag-and-drop reordering.
-   **Real-time Preview**: Instantly view "Before" and "After" data snapshots (`Head(10)`).
-   **Error Resolution**: AI-driven error diagnosis and resolution suggestions for failed transformations.

### 3. Visual Intelligence Studio
-   **Interactive Dashboards**: Create and view visualizations of your data.
-   **Dynamic Charting**: (Feature in progress) Generate charts to analyze trends and patterns.

### 4. Agent Studio
-   **LangChain Integration**: specialized environment for running and managing LangChain agents.
-   **Workflow Automation**: Execute complex, multi-step data tasks using autonomous agents.

---

## 🛠️ Tech Stack

-   **Backend**: Django 5, Python 3.12
-   **Data Processing**: Pandas, NumPy, Scikit-learn, SQLAlchemy
-   **AI & LLM**: Ollama (Local LLM Inference), LangChain
-   **Frontend**: Django Templates, TailwindCSS, React (embedded)
-   **Database**: SQLite (default), PostgreSQL (connector supported)

---

## ⚡ Getting Started

### Prerequisites
-   Python 3.10+
-   [Ollama](https://ollama.com/) (for AI features)
-   Git

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/priyamghosh0412/NeuralFlow-Studio.git
    cd NeuralFlow-Studio
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Setup AI Service**
    -   Install Ollama from [ollama.com](https://ollama.com).
    -   Pull the required model (default: `codellama:7b`):
        ```bash
        ollama pull codellama:7b
        ```
    -   Start the Ollama server:
        ```bash
        ollama serve
        ```

5.  **Run Migrations**
    ```bash
    python manage.py migrate
    ```

6.  **Start the Server**
    ```bash
    python manage.py runserver
    ```
    Access the application at `http://127.0.0.1:8000/`.

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License.
