# Quick Start Guide

## Local Development

1. **Run setup script**:
   ```bash
   ./setup.sh
   ```

2. **Start the app**:
   ```bash
   conda activate askmydocs
   streamlit run app.py
   ```

3. **Open browser**: http://localhost:8501

## GitHub Setup

1. **Create GitHub repo** at https://github.com/new
2. **Push your code**:
   ```bash
   git remote add origin https://github.com/yourusername/askmydocs.git
   git branch -M main
   git push -u origin main
   ```

## Render Deployment

1. **Go to Render.com** and sign in
2. **Create Web Service** from your GitHub repo
3. **Use these settings**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
   - Plan: Free

4. **Wait for deployment** and test your live app!

## Sample Documents

- `test_document.txt` - Simple story for testing
- `Trends_Artificial_Intelligence.pdf` - AI research report

## Need Help?

- Check `DEPLOYMENT.md` for detailed instructions
- View Render logs for deployment issues
- Test with small files first on free tier