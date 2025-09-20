# 🚀 Machine Learning Pipeline with Kaggle Integration

A comprehensive machine learning application built with Streamlit that integrates multiple data sources including Kaggle API for seamless dataset access and ML model training.

## ✨ Features

### 📊 **Data Sources**
- **Upload CSV**: Upload your own datasets
- **OpenML**: Access 1000+ research datasets
- **Hugging Face**: NLP and text datasets
- **🏆 Kaggle**: Direct access to popular ML competition datasets

### 🤖 **Machine Learning**
- **Classification & Regression**: Automatic problem type detection
- **Multiple Algorithms**: Logistic Regression, Random Forest, Linear Regression
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Predictions**: Manual input and batch prediction capabilities

### 📈 **Exploratory Data Analysis**
- Correlation heatmaps
- Feature distribution plots
- Statistical summaries
- Missing value analysis

### 🎯 **Kaggle Integration Features**
- **Popular Datasets**: Pre-selected popular ML datasets
- **Search Functionality**: Search Kaggle's vast dataset collection
- **Manual Input**: Direct dataset reference input
- **Automatic Processing**: Seamless CSV extraction and processing

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ml-pipeline
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run ml_app.py
   ```

4. **Open your browser** and go to `http://localhost:8501`

## 🏆 Kaggle Integration Usage

### 1. **Popular Datasets**
Select from pre-configured popular datasets:
- Titanic - Machine Learning from Disaster
- House Prices - Advanced Regression Techniques
- Credit Card Fraud Detection
- Iris Species
- Wine Quality
- And more...

### 2. **Search Kaggle**
Search for datasets by keywords:
```
"titanic" → Survival prediction datasets
"housing" → Real estate datasets  
"fraud" → Fraud detection datasets
"sentiment" → Text sentiment analysis
```

### 3. **Manual Input**
Enter Kaggle dataset references directly:
```
c/titanic
c/house-prices-advanced-regression-techniques
uciml/iris
```

## 📁 Project Structure

```
ml-pipeline/
├── ml_app.py                    # Main Streamlit application
├── simple_ml_app.py            # Simplified version
├── app (1).py                  # Alternative app structure
├── datasets_search (1).py      # Dataset search utilities
├── models (1).py               # Model definitions
├── preprocessing (1).py        # Data preprocessing
├── utils (1).py                # Utility functions
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── kaggle_integration_demo.py  # Integration documentation
└── README.md                   # This file
```

## 🔧 Technical Details

### **Kaggle API Integration**
- **Authentication**: Secure credential management
- **Download**: Direct dataset download with automatic CSV extraction
- **Error Handling**: Comprehensive error handling and user feedback
- **Memory Management**: Efficient temporary file handling

### **Supported Dataset Types**
- CSV files (automatically detected)
- Competition datasets
- User-uploaded datasets
- Public datasets

### **ML Pipeline Features**
- **Preprocessing**: Automatic data cleaning and encoding
- **Train/Test Split**: 80/20 split with random state
- **Model Training**: Multiple algorithms with hyperparameter tuning
- **Evaluation**: Detailed metrics and visualizations
- **Predictions**: Real-time prediction interface

## 📊 Example Workflow

1. **Select Data Source**: Choose "Fetch from Kaggle"
2. **Download Dataset**: Pick from popular datasets or search
3. **Select Target**: Choose the column to predict
4. **EDA**: Explore data with built-in visualizations
5. **Train Models**: Automatic model training and evaluation
6. **Make Predictions**: Use trained models for new predictions
7. **Export Results**: Download predictions and model results

## 🛠️ Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library
- **matplotlib**: Plotting library
- **seaborn**: Statistical data visualization
- **openml**: OpenML dataset access
- **datasets**: Hugging Face datasets
- **kaggle**: Kaggle API client
- **scipy**: Scientific computing

## 🔒 Security Features

- Temporary credential storage
- Automatic cleanup of sensitive data
- Secure API communication
- No persistent credential storage

## 📈 Performance Optimizations

- Largest CSV file selection for multi-file datasets
- Efficient memory usage
- Background processing with progress indicators
- Automatic temporary file cleanup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter any issues:
1. Check the error messages in the application
2. Ensure all dependencies are installed
3. Verify Kaggle API credentials (if using Kaggle features)
4. Check the console output for detailed error information

## 🎯 Future Enhancements

- [ ] Support for more ML algorithms
- [ ] Hyperparameter tuning interface
- [ ] Model persistence and loading
- [ ] Advanced feature engineering
- [ ] Real-time model monitoring
- [ ] Integration with cloud ML platforms

---

**Happy Machine Learning! 🎉**
