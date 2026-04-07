# Deployment Guide - AskMyDocs

This guide covers deploying AskMyDocs to Render.com.

## Prerequisites

- GitHub account with the AskMyDocs repository
- Render.com account
- Ollama installed locally (for development)

## Step 1: Push to GitHub

1. **Create a new repository** on GitHub
2. **Add remote origin** to your local repository:
   ```bash
   git remote add origin https://github.com/yourusername/askmydocs.git
   git branch -M main
   git push -u origin main
   ```

## Step 2: Deploy to Render

1. **Sign in to Render.com** and connect your GitHub account

2. **Create a new Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository: `yourusername/askmydocs`

3. **Configure the service**:
   - **Name**: `askmydocs` (or your preferred name)
   - **Environment**: `Python`
   - **Region**: Choose closest to your users
   - **Branch**: `main`
   - **Root Directory**: (leave empty)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
   - **Plan**: `Free` (or upgrade for better performance)

4. **Advanced Settings** (optional):
   - **Environment Variables**: No additional variables needed for basic setup
   - **Auto-Deploy**: Enable for automatic deployments on git push

5. **Create Service**: Click "Create Web Service"

## Step 3: Verify Deployment

1. **Wait for build** to complete (5-10 minutes)
2. **Check logs** for any errors
3. **Open your app** at the provided Render URL
4. **Test functionality** by uploading a document and asking questions

## Important Notes

### Ollama on Render
- Render free tier doesn't support Ollama installation
- The app uses Ollama's cloud models (`deepseek-v3.1:671b-cloud`, `minimax-m2:cloud`, `glm-4.6:cloud`)
- No additional setup required for cloud models

### File Size Limitations
- Render free tier has memory limitations
- Large PDFs (>10MB) may cause memory issues
- Consider upgrading plan for production use

### Performance Considerations
- Free tier may have cold start delays
- TF-IDF processing is done in memory
- For large documents, consider chunk size optimization

## Troubleshooting

### Common Issues

1. **Build fails**: Check requirements.txt compatibility
2. **App crashes**: Check Render logs for memory errors
3. **Ollama errors**: Ensure cloud models are accessible
4. **PDF extraction issues**: PyMuPDF may have limitations with some PDFs

### Logs Access
- View logs in Render dashboard under your service
- Check both build logs and runtime logs

## Scaling

For production use:
1. Upgrade to paid plan for better resources
2. Consider adding Redis for caching
3. Implement database for document storage
4. Add user authentication if needed

## Monitoring

- Render provides basic monitoring
- Consider adding health checks
- Monitor memory usage for large documents