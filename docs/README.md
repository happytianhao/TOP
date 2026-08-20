# TOP Project Page

This directory contains the project page for TOP (Temporal Offset Prediction for Accident Anticipation).

## Viewing Locally

To view the project page locally, simply open `index.html` in a web browser:

```bash
cd docs
python -m http.server 8000
# Then visit http://localhost:8000
```

## GitHub Pages Deployment

To deploy this page on GitHub Pages:

1. Push the repository to GitHub
2. Go to repository Settings → Pages
3. Set Source to "Deploy from a branch"
4. Select branch: `main` and folder: `/docs`
5. Save and wait for deployment

The page will be available at: `https://YOUR_USERNAME.github.io/TOP/`

## Customization

### Update Results
Edit the metric values in `index.html` (search for "XX.X%") with your actual results.

### Add Demo Video
If you have a demo video, add it by inserting this code in the appropriate section:

```html
<div class="video-container">
    <iframe src="https://www.youtube.com/embed/YOUR_VIDEO_ID" 
            frameborder="0" 
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
            allowfullscreen>
    </iframe>
</div>
```

### Update Author Information
Replace "Anonymous Authors" with actual author names and affiliations when ready.

### Update GitHub Links
Replace `YOUR_USERNAME` with your actual GitHub username throughout the page.
