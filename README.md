# LegalBot Setup Guide

This guide walks you through setting up and running the LegalBot project locally on macOS.

## Setup Steps

### Step 1: Create a Virtual Environment

```bash
python3 -m venv venv
```

### Step 2: Activate the Virtual Environment

```bash
source venv/bin/activate
```

### Step 3: Install Required Dependencies

If you have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 4: Create a .env File

Create a file named `.env` in the root directory and add the following:

```
PINECONE_API=your-pinecone-api-key
OPENAI_API=your-openai-api-key
```

### Step 5: Run the App

```bash
python app.py
```

You can now send a POST request to http://127.0.0.1:5000/legalchat with a JSON body like:

```json
{
  "query": "My client, a Black male, was found with a loaded handgun in his car. He is a previously convicted felon, but the firearm was discovered during a traffic stop. What defense strategies have worked in similar cases, and what sentencing outcomes can I expect?"
}
```