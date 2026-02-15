<h1>📝 SmartQGen: AI-Powered Question Paper Generator</h1>
SmartQGen is an intelligent web application designed to automate the process of creating high-quality question papers from raw text or PDF documents. By leveraging Google's Gemini-1.5-Pro/2.0-Flash models, it transforms dense educational content into structured assessments, including MCQs, theoretical Q&A, and fill-in-the-blanks.

<h2>🚀 Unique Value Proposition</h2>
Traditional paper setting is time-consuming and prone to human bias. SmartQGen stands out by:

Contextual Intelligence: Unlike simple randomizers, it uses Generative AI to understand the semantic meaning of your text to create relevant problems.

Specialized Physics Engine: Includes custom logic for analyzing physics texts to extract formulas and key concepts without LaTeX distortions.

Dynamic Difficulty Scaling: Allows users to generate "Hard" or "Medium" problems based on the depth of the source material.

Deployment Ready: Architected as a serverless-friendly Flask application, optimized for instant scaling on Vercel.

<h2>🛠️ Tech Stack</h2>
Backend: Python / Flask

AI Orchestration: Google Generative AI (Gemini API)

Database: SQLAlchemy (SQLite) for user session management

Document Processing: pdfplumber and PyPDF2 for robust extraction

Frontend: HTML5, Jinja2 Templates, and CSS

Deployment: Vercel (Serverless Functions)

<h2>📋 Key Features</h2>
Multi-Format Extraction: Support for PDF, TXT, and DOCX files.

Automated Quiz Generation: Instantly creates MCQs with correct answer tracking.

Theoretical Q&A: Generates deep-dive questions and comprehensive solutions.

Physics-Specific Extraction: Identifies and filters questions, answers, and formulas specifically from scientific texts.

Export to PDF: Converts generated question papers into professional PDF documents for offline use.

<h2>⚙️ Installation & Setup</h2>
Prerequisites
Python 3.9+
Google Gemini API Key
codes:
pip install -r requirements.txt,
python home_app.py
