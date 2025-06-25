
# Multi-Agent Data Analytics & Visualization Platform
 
A powerful Streamlit-based application that uses specialized AI agents to analyze data and create visualizations through natural language queries.
 
## Features
 
- **Multi-Agent Architecture**: Specialized agents for data analysis, visualization, and conversational queries
- **File Support**: Upload and analyze CSV, DOC, DOCX, and PDF files
- **Intelligent Analytics**: Automated data quality assessment and insight generation
- **Natural Language Visualization**: Create charts using plain English descriptions
- **Interactive Chat**: Ask questions about your data in natural language
- **Real-time Processing**: Live data analysis and visualization generation
 
## Project Structure
 
```
DataInsightsAI/
├── agents/
│   ├── data_analytics_agent.py # Data analysis and insights
│   ├── gemini_agent.py #LLM
│   └── visualization_agent.py  # Chart generation
├── utils/
│   ├── chart_generator.py      # Plotly chart creation utilities
│   ├── data_handler.py         # Data preprocessing utilities
│   |── file_processor.py       # File upload and processing
├── .streamlit/
    |── config.toml             # Streamlit configuration
├── app.py                      # Main application entry point
├── dual_rag_agent_system.py        #Dual Rag Support
├── pyproject.toml              # Python dependencies
|── README.md                   # This file
```
 
## Installation
 
1. **Extract the project files:**
   ```bash
   tar -xzf multi_agent_analytics_app.tar.gz
   cd multi-agent-analytics
   ```
 
2. **Install Python dependencies:**
   ```bash
   pip install streamlit pandas plotly pypdf2 python-docx scikit-learn spacy textblob
   ```
 
3. **Install spaCy language model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```
 
## Usage
 
1. **Start the application:**
   ```bash
   streamlit run app.py
   ```
 
2. **Access the web interface:**
   - Open your browser to `http://localhost:8501`
 
3. **Upload and analyze data:**
   - Use the sidebar to upload CSV, DOC, DOCX, or PDF files
   - Navigate through the tabs to explore different features
 
## Application Tabs
 
### 📊 Data Preview
- View raw data and basic statistics
- Examine data types and missing values
- Quick overview of dataset structure
 
### 🔍 Analytics
- Generate comprehensive data analysis reports
- Data quality assessment with completeness, consistency, and validity metrics
- AI-powered insights and recommendations
 
### 📈 Visualizations
- Create charts using natural language queries
- Examples: "Create a bar chart showing sales by region"
- Quick visualization options for common chart types
 
### 💬 Chat Interface
- Interactive conversations about your data
- Ask questions in plain English
- Get insights and explanations from AI agents
 
## Agent Capabilities
 
### Data Analytics Agent
- Automated data cleaning and preprocessing
- Statistical analysis and pattern detection
- Data quality assessment
- Insight generation and recommendations
 
### Visualization Agent
- Natural language chart generation
- Support for multiple chart types (bar, line, scatter, pie, etc.)
- Intelligent column mapping and configuration
- Interactive Plotly visualizations
 
### Conversational Agent (Gemma-powered)
- Advanced natural language understanding
- Context-aware responses
- Multi-turn conversations about data
- Intelligent query routing
 
## Supported File Types
 
- **CSV**: Tabular data for statistical analysis
- **DOC/DOCX**: Microsoft Word documents for text analysis
- **PDF**: Portable Document Format for content extraction
 
## Configuration
 
The application can be customized through `.streamlit/config.toml`:
 
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```
 
## Dependencies
 
Core packages:
- `streamlit`: Web application framework
- `pandas`: Data manipulation and analysis
- `plotly`: Interactive visualizations
- `scikit-learn`: Machine learning utilities
- `spacy`: Natural language processing
- `textblob`: Text processing and sentiment analysis
- `pypdf2`: PDF file processing
- `python-docx`: Word document processing
 
## Development
 
To extend the application:
 
1. **Add new agents**: Create new agent classes in the `agents/` directory
2. **Extend file support**: Modify `utils/file_processor.py` for new file types
3. **Add chart types**: Extend `utils/chart_generator.py` with new visualization options
4. **Customize styling**: Modify the Streamlit configuration and CSS
 
## Troubleshooting
 
**Common Issues:**
 
1. **Import errors**: Ensure all dependencies are installed
2. **File upload failures**: Check file format and size limitations
3. **Visualization errors**: Verify data types and column names
4. **Performance issues**: Consider data sampling for large datasets
 
**Performance Tips:**
 
- Use CSV files for optimal performance with large datasets
- Limit text analysis to essential portions of large documents
- Consider data preprocessing for complex datasets
 
## License
 
This project is provided as-is for educational and development purposes.
 
## Support
 
For issues and questions, please refer to the application's built-in help system or check the console output for error messages.
 