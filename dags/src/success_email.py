import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from jinja2 import Template
from airflow.hooks.base import BaseHook
import json
import os

def send_success_email(**kwargs):
    """
    Send success email with model evaluation results.
    """
    try:
        # Get SMTP credentials from Airflow connection or environment
        try:
            conn = BaseHook.get_connection('email_smtp')
            sender_email = conn.login
            password = conn.password
        except:
            # Fallback to environment variables
            sender_email = os.environ.get('SMTP_USER')
            password = os.environ.get('SMTP_PASSWORD')
        
        if not sender_email or not password:
            print("ERROR: SMTP credentials not found!")
            return
        
        # Receiver email - UPDATE THIS WITH YOUR EMAIL
        receiver_email = 'patel.malav@northeastern.edu'  # CHANGE THIS
        
        # Load evaluation results
        results_path = "/opt/airflow/model/evaluation_results.json"
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
        else:
            results = {'accuracy': 'N/A', 'precision': 'N/A', 'recall': 'N/A', 'f1_score': 'N/A'}
        
        # Email subject
        subject = f"✅ Airflow Success: {kwargs['dag'].dag_id} - Healthcare ML Pipeline Completed"
        
        # Email body with results
        body_template = '''
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #4CAF50;">🎉 ML Pipeline Execution Successful!</h2>
            
            <p>Hi Malav,</p>
            
            <p>The healthcare ML pipeline in DAG <strong>{{ dag.dag_id }}</strong> has completed successfully.</p>
            
            <h3> Model Performance Metrics:</h3>
            <table style="border-collapse: collapse; width: 50%;">
                <tr style="background-color: #f2f2f2;">
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Metric</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Value</th>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Accuracy</td>
                    <td style="border: 1px solid #ddd; padding: 8px;"><strong>{{ accuracy }}</strong></td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Precision</td>
                    <td style="border: 1px solid #ddd; padding: 8px;"><strong>{{ precision }}</strong></td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Recall</td>
                    <td style="border: 1px solid #ddd; padding: 8px;"><strong>{{ recall }}</strong></td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">F1-Score</td>
                    <td style="border: 1px solid #ddd; padding: 8px;"><strong>{{ f1_score }}</strong></td>
                </tr>
            </table>
            
            <h3>🔝 Top Features:</h3>
            <ul>
            {% for feature, importance in top_features.items() %}
                <li><strong>{{ feature }}</strong>: {{ importance }}</li>
            {% endfor %}
            </ul>
            
            <p style="margin-top: 20px;">
                <strong>Execution Time:</strong> {{ execution_date }}<br>
                <strong>Task:</strong> {{ task.task_id }}
            </p>
            
            <p style="color: #666; font-size: 12px; margin-top: 30px;">
                This is an automated email from your Airflow pipeline.<br>
                Northeastern University - CS Department
            </p>
        </body>
        </html>
        '''
        
        # Render template
        body = Template(body_template).render(
            dag=kwargs['dag'],
            task=kwargs['task'],
            execution_date=kwargs['execution_date'],
            accuracy=results.get('accuracy', 'N/A'),
            precision=results.get('precision', 'N/A'),
            recall=results.get('recall', 'N/A'),
            f1_score=results.get('f1_score', 'N/A'),
            top_features=results.get('top_features', {})
        )
        
        # Create email
        email_message = MIMEMultipart('alternative')
        email_message['Subject'] = subject
        email_message['From'] = sender_email
        email_message['To'] = receiver_email
        
        # Attach HTML body
        email_message.attach(MIMEText(body, 'html'))
        
        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, email_message.as_string())
        server.quit()
        
        print(f"✅ Success email sent to {receiver_email}!")
        
    except Exception as e:
        print(f"❌ Error sending success email: {e}")
        import traceback
        traceback.print_exc()