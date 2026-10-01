"""
build.py
Generates a complete, production-ready static site bundle in the `dist/` directory
optimized for 100% serverless deployment to Netlify (as well as GitHub Pages, Vercel, or Cloudflare Pages).
"""

import os
import shutil
import subprocess
import sys


def build_static_site():
    print("=" * 60)
    print("🚀 Building Static Site for Netlify Deployment")
    print("=" * 60)

    # 1. Export the latest machine learning JavaScript engine
    print("\n📦 Step 1: Exporting client-side ML engine...")
    import export_engine
    export_engine.export_model()

    # 2. Initialize Flask app test client
    print("\n🌐 Step 2: Rendering static HTML pages from Flask application...")
    from app import app
    app.config["TESTING"] = True
    client = app.test_client()

    dist_dir = os.path.abspath("dist")
    if os.path.exists(dist_dir):
        shutil.rmtree(dist_dir)
    os.makedirs(dist_dir, exist_ok=True)

    routes = [
        ("/", "index.html"),
        ("/predict", "predict.html"),
        ("/about", "about.html"),
        ("/result", "result.html")
    ]

    for route, filename in routes:
        response = client.get(route)
        if response.status_code != 200:
            print(f"❌ Failed to render {route} (HTTP {response.status_code})")
            sys.exit(1)

        html_content = response.data.decode("utf-8")
        out_path = os.path.join(dist_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"   ✓ Rendered {route:12} -> dist/{filename} ({len(html_content):,} bytes)")

        # Also create directory index for pretty URLs: e.g. dist/predict/index.html
        if filename != "index.html":
            dir_name = os.path.splitext(filename)[0]
            sub_dir = os.path.join(dist_dir, dir_name)
            os.makedirs(sub_dir, exist_ok=True)
            with open(os.path.join(sub_dir, "index.html"), "w", encoding="utf-8") as f:
                f.write(html_content)

    # 3. Copy static directory
    print("\n📁 Step 3: Copying static assets (CSS, JS, images, logos)...")
    dist_static = os.path.join(dist_dir, "static")
    shutil.copytree("static", dist_static)
    static_files_count = sum(len(files) for _, _, files in os.walk(dist_static))
    print(f"   ✓ Copied {static_files_count} static assets to dist/static/")

    # 4. Generate Netlify _redirects file
    print("\n🔀 Step 4: Generating Netlify routing configuration (_redirects)...")
    redirects_content = """# Netlify Redirects & Clean URLs
/predict   /predict.html   200
/about     /about.html     200
/result    /result.html    200
/*         /index.html     200
"""
    with open(os.path.join(dist_dir, "_redirects"), "w", encoding="utf-8") as f:
        f.write(redirects_content)
    print("   ✓ Created dist/_redirects")

    # 5. Generate Netlify _headers file
    print("\n🛡️ Step 5: Generating Netlify security and caching headers (_headers)...")
    headers_content = """/*
  X-Frame-Options: SAMEORIGIN
  X-Content-Type-Options: nosniff
  Referrer-Policy: strict-origin-when-cross-origin
  Permissions-Policy: camera=(), microphone=(), geolocation=()

/static/*
  Cache-Control: public, max-age=31536000, immutable
"""
    with open(os.path.join(dist_dir, "_headers"), "w", encoding="utf-8") as f:
        f.write(headers_content)
    print("   ✓ Created dist/_headers")

    print("\n" + "=" * 60)
    print("🎉 Static site build COMPLETE! Directory ready: dist/")
    print("   Deploy to Netlify via:")
    print("     1. Netlify CLI: npx netlify deploy --prod --dir=dist")
    print("     2. Netlify Git: push to GitHub and set publish directory to 'dist'")
    print("     3. Netlify UI: drag & drop the 'dist' directory into app.netlify.com")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    build_static_site()
