# Netlify Deployment Guide

## Hepatitis C Detection and Staging using Machine Learning
*AI for Healthcare: Mini Project*

This repository is fully configured and optimized for instant, zero-configuration deployment to **Netlify**.

---

## Architecture Overview

While traditional machine learning applications require continuous Python backend servers, this project features a **hybrid dual-mode architecture**:

1. **Local & Cloud Backend (Flask)**:
   - When run locally (`python app.py`) or on platforms like Render / Railway, full server-side inference and dynamic routes are active.
2. **Netlify Static & Client-Side ML Engine**:
   - For Netlify, the application pre-renders static HTML pages (`index.html`, `predict.html`, `about.html`, `result.html`) into the `dist/` directory via `build.py`.
   - The trained **XGBoost gradient boosting ensemble** (all 1,500 decision trees, median imputers, and standard scalers) is compiled into a lightweight client-side JavaScript engine (`static/js/hcv_engine.js`, ~200 KB).
   - Predictions run directly in the user's browser in **<1ms**, requiring zero backend server compute, zero cold starts, and 100% offline capability.

---

## Deployment Methods

### Option 1: Git-Connected Continuous Deployment (Recommended)

1. Log in to [Netlify](https://app.netlify.com/).
2. Click **"Add new site"** → **"Import an existing project"**.
3. Choose **GitHub** and select your repository: `AIH-Hepatitis-C---Mini-Project`.
4. Netlify will automatically detect [`netlify.toml`](netlify.toml) with the correct settings:
   - **Build command**: `python build.py`
   - **Publish directory**: `dist`
5. Click **"Deploy site"**.
6. Every time you push to `main`, Netlify automatically rebuilds and deploys the latest version!

---

### Option 2: Drag & Drop Instant Deploy (No CLI or Git Required)

The repository comes with the production bundle pre-built in `dist/`.

1. Run `python build.py` locally to make sure the latest build is present:
   ```bash
   python build.py
   ```
2. Navigate to [app.netlify.com/drop](https://app.netlify.com/drop).
3. Drag and drop the `dist` folder directly from your file manager into the upload box.
4. Your site is live on a custom `.netlify.app` URL within 5 seconds!

---

### Option 3: Netlify CLI

You can deploy directly from your terminal using `npx` (no global installation needed):

```bash
# 1. Build the production bundle
python build.py

# 2. Deploy directly to Netlify production
npx netlify deploy --prod --dir=dist
```

---

## Netlify Configuration Reference

The project includes pre-configured Netlify files:

- **[`netlify.toml`](netlify.toml)**: Defines build commands, Python 3.11 environment, pretty URL routing, and security headers.
- **`dist/_redirects`**: Handles single-page and multi-page routing:
  ```text
  /predict   /predict.html   200
  /about     /about.html     200
  /result    /result.html    200
  /*         /index.html     200
  ```
- **`dist/_headers`**: Provides security headers (`X-Frame-Options`, `X-Content-Type-Options`, `Referrer-Policy`) and asset caching rules.

---

## Optional: Gemini Vision AI Key for Lab Report Scanning

If you want the physical report OCR scanner to work on Netlify:
1. In your Netlify Site dashboard, go to **Site configuration** → **Environment variables**.
2. Add:
   - **Key**: `GEMINI_API_KEY`
   - **Value**: Your Google AI Studio API key (`AIzaSy...`)
3. Users can also enter their key directly in the browser UI via the settings gear on the Predict page.
