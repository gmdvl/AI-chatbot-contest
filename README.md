# Enhanced STEM Tutor Bot - Setup & Running Guide

## Quick Start

Follow these steps to run the STEM Tutor Bot on your computer.

---

## Prerequisites

Before starting, make sure you have:

- **Python 3.8 or higher** installed
- **pip** (Python package installer)
- **Internet connection** (for downloading models and datasets)
- **At least 5-10 GB free disk space** (for AI models)
- **4-8 GB RAM** recommended

### Check Python Version

Open PowerShell or Command Prompt and run:

```powershell
python --version
```

You should see something like `Python 3.8.x` or higher.

---

## Step 1: Install Dependencies

### Option A: Install All Dependencies (Recommended)

Open PowerShell in the project folder and run:

```powershell
pip install Flask transformers torch datasets sentence-transformers numpy
```

**Note:** This may take 10-20 minutes as it downloads large AI models.

### Option B: Install from requirements.txt

If you want to install all packages from the requirements file:

```powershell
pip install -r requirements.txt
pip install Flask
```

**Important:** Flask is required but may not be in requirements.txt, so install it separately if needed.

---

## Step 2: Run the Application

### Start the Flask Server

Open PowerShell in the project folder and run:

```powershell
python app.py
```

### What Happens Next:

1. **Model Loading** (2-5 minutes on first run):
   - The bot will download and load AI models
   - You'll see progress messages like:
     ```
     🔧 Initializing AI models and datasets...
     📊 Loading semantic search model...
     ✅ Semantic model loaded!
     🤖 Loading BERT QA model...
     ✅ BERT model loaded!
     📚 Loading datasets...
     ```

2. **Server Starts**:
   - You'll see: `Running Flask STEM Tutor Bot...`
   - The server will be available at: `http://127.0.0.1:5000` or `http://localhost:5000`

3. **Open in Browser**:
   - Open your web browser
   - Go to: `http://localhost:5000`
   - You should see the chat interface!

---

## Step 3: Use the Chatbot

### In the Browser:

1. **Wait for "Ready" status** (green indicator in top-right)
2. **Type a question** in the input box, for example:
   - "What is Newton's second law?"
   - "Explain covalent bonding"
   - "What is photosynthesis?"
   - "How do I solve quadratic equations?"

3. **Click "Send"** or press Enter
4. **Wait for response** (status shows "Thinking..." while processing)
5. **Read the answer** with confidence score

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'flask'"

**Solution:**
```powershell
pip install Flask
```

### Problem: "ModuleNotFoundError: No module named 'transformers'"

**Solution:**
```powershell
pip install transformers torch datasets sentence-transformers numpy
```

### Problem: Models take too long to download

**Solution:**
- This is normal on first run (models are 1-5 GB)
- Ensure stable internet connection
- Wait patiently - it only happens once

### Problem: "Port 5000 already in use"

**Solution:**
- Close other applications using port 5000
- Or modify `app.py` line 75 to use a different port:
  ```python
  app.run(debug=True, threaded=False, port=5001)
  ```

### Problem: Out of memory errors

**Solution:**
- Close other applications
- Reduce dataset sampling in `bert.py`:
  - Line 627: Change `if i > 3000:` to `if i > 1000:`
  - Line 858: Change `if i > 2000:` to `if i > 1000:`

### Problem: Browser shows "Connection refused"

**Solution:**
- Make sure Flask server is running (check PowerShell window)
- Wait for "Running Flask STEM Tutor Bot..." message
- Try `http://127.0.0.1:5000` instead of `localhost:5000`

---

## File Structure

```
AI-chatbot-contest-master/
├── app.py                 # Flask web server (run this!)
├── bert.py                # Bot AI logic
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── PROJECT_SUMMARY.md    # Project documentation
└── templates/
    └── index.html        # Web interface
```

---

## Command Reference

### Start the server:
```powershell
python app.py
```

### Stop the server:
- Press `Ctrl + C` in the PowerShell window

### Install missing packages:
```powershell
pip install [package-name]
```

### Check if Flask is installed:
```powershell
python -c "import flask; print(flask.__version__)"
```

---

## Expected Behavior

### First Run:
- **Loading time:** 2-5 minutes (downloading models)
- **Memory usage:** 2-4 GB RAM
- **Disk space:** 5-10 GB (models cached after first run)

### Subsequent Runs:
- **Loading time:** 30 seconds - 2 minutes (models cached)
- **Faster startup** once models are downloaded

### Chat Response Times:
- **Knowledge Base:** 0.5-1 second
- **Dataset Search:** 1-4 seconds
- **Total:** Usually 1-5 seconds per question

---

## Testing the Bot

Try these sample questions to test different features:

1. **Physics:**
   - "What is Newton's first law?"
   - "Explain kinetic energy"
   - "What is gravity?"

2. **Chemistry:**
   - "What is an atom?"
   - "Explain covalent bonding"
   - "What is the pH scale?"

3. **Biology:**
   - "What is photosynthesis?"
   - "Explain cellular respiration"
   - "What is DNA?"

4. **Mathematics:**
   - "What is the quadratic formula?"
   - "Explain the Pythagorean theorem"

---

## Advanced: Running in Background

### Windows PowerShell (Background):
```powershell
Start-Process python -ArgumentList "app.py" -WindowStyle Hidden
```

### Or use a new terminal:
- Keep PowerShell running the server
- Open a new terminal for other commands

---

## Need Help?

If you encounter issues:

1. **Check Python version:** `python --version` (need 3.8+)
2. **Check installed packages:** `pip list`
3. **Read error messages** in the PowerShell window
4. **Check internet connection** (needed for first-time model download)
5. **Restart the server** (Ctrl+C, then `python app.py` again)

---

## Summary

**To run the bot:**

1. Open PowerShell in project folder
2. Run: `pip install Flask transformers torch datasets sentence-transformers numpy`
3. Run: `python app.py`
4. Wait for models to load (first time: 2-5 minutes)
5. Open browser: `http://localhost:5000`
6. Start chatting!

**That's it!** 🎉

