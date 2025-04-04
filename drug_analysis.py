import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, 
                           roc_curve, auc, precision_recall_curve)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class DrugConsumptionAnalysis:
    def __init__(self):
        # Fetch dataset
        self.dataset = fetch_ucirepo(id=373)
        self.X = self.dataset.data.features
        self.y = self.dataset.data.targets
        self.drug_names = list(self.y.columns)
        self.features = list(self.X.columns)
        
        # Initialize encoders and scalers
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def preprocess_data(self, target_drug='Cannabis'):
        # Encode the target variable
        self.label_encoders[target_drug] = LabelEncoder()
        y_encoded = self.label_encoders[target_drug].fit_transform(self.y[target_drug])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(self.X)
        
        return X_scaled, y_encoded
    
    def train_model(self, X, y):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        return rf_model, X_train, X_test, y_train, y_test
    
    def evaluate_model(self, model, X_test, y_test):
        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        
        return {
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'roc_data': (fpr, tpr, roc_auc),
            'pr_data': (precision, recall)
        }
    
    def create_feature_importance_plot(self, model):
        importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h'
        ))
        
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=800
        )
        
        return fig
    
    def create_correlation_heatmap(self):
        corr_matrix = self.X.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=self.features,
            y=self.features,
            colorscale='RdBu'
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            height=800,
            width=800
        )
        
        return fig
    
    def create_drug_usage_distribution(self):
        fig = make_subplots(rows=2, cols=2)
        
        for idx, drug in enumerate(self.drug_names[:4]):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            value_counts = self.y[drug].value_counts()
            
            fig.add_trace(
                go.Bar(x=value_counts.index, y=value_counts.values, name=drug),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text=drug, row=row, col=col)
            fig.update_yaxes(title_text='Count', row=row, col=col)
        
        fig.update_layout(
            title='Drug Usage Distribution',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_personality_distribution(self):
        personality_traits = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore']
        
        fig = go.Figure()
        
        for trait in personality_traits:
            fig.add_trace(go.Box(
                y=self.X[trait],
                name=trait,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
        
        fig.update_layout(
            title='Distribution of Personality Traits',
            yaxis_title='Score',
            height=600
        )
        
        return fig
