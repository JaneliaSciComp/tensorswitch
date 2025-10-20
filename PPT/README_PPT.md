# TensorSwitch Presentation

Dynamic presentation built with impress.js showcasing the TensorSwitch journey from scattered scripts to production-ready tool.

## Files Structure

```
PPT/
├── Diyi_tensorswitch_presentation.html    # Main presentation file
├── README_PPT.md                           # This file
├── css/
│   ├── impress.css                   # Impress.js base styles
│   ├── presentation-style.css        # Custom TensorSwitch styles
│   └── impressConsole.css            # Speaker console styles
├── js/
│   ├── impress.js                    # Impress.js library
│   └── impressConsole.js             # Speaker console script
└── images_videos/                    # Your images and videos go here
    └── (add your media files here)
```

## How to View the Presentation

### Quick Start

1. Open the presentation in a web browser:
   ```bash
   cd /groups/scicompsoft/home/chend/temp/downsample_script/tensorswitch/PPT
   firefox Diyi_tensorswitch_presentation.html
   # or
   google-chrome Diyi_tensorswitch_presentation.html
   ```

2. **No installation required!** Everything is self-contained (no Python/Node packages needed).

### Navigation

- **Arrow Keys**:
  - **←/→**: Move through timeline horizontally
  - **↑/↓**: Dive into details (↓) or return to timeline (↑)
  - **Space**: Next slide
  - **Home**: First slide
  - **End**: Last slide

- **Special Keys**:
  - **P**: Open speaker notes/presenter view (opens in new window)
  - **O**: Overview mode (see all slides at once)

### Presenter Mode

1. Press **P** to open the speaker console (requires allowing pop-ups)
2. The console shows:
   - **Timer**: Elapsed presentation time
   - **Current Slide**: What the audience sees
   - **Next Slide**: What's coming next
   - **Speaker Notes**: Detailed notes for you to read (NOT visible to audience)

3. Navigate normally in the main window - the console updates automatically

## Adding Images and Videos

### Image Placeholders

The HTML has commented placeholders like:
```html
<!-- <img src="images_videos/tensorswitch_logo.png" alt="TensorSwitch Logo" class="logo"> -->
```

To add an image:
1. Save your image to `images_videos/` folder
2. Uncomment the line and update the filename
3. Refresh the presentation

### Video Placeholders

Similar for videos:
```html
<!-- <video controls src="images_videos/cli_demo.mp4" class="demo-video"></video> -->
```

### Supported Formats

- **Images**: PNG, JPG, GIF, SVG
- **Videos**: MP4, WebM, OGG

## Presentation Structure

The presentation follows a timeline with 7 phases:

1. **Origin**: Where TensorSwitch came from
   - Name etymology (TensorStore + Switch)
   - Lab requests that sparked development

2. **Development**: Technical foundation
   - Format differences (N5, Zarr2, Zarr3, IMS, ND2, TIFF)
   - The 10 conversion tasks

3. **Personal Tool**: Organization phase
   - CLI workflow improvements

4. **GUI Born**: Empowering scientists
   - Smart Mode with auto-detection
   - Manual Mode for power users
   - Lab paths integration
   - Cost estimation

5. **AI Power**: Intelligent assistance
   - Q&A examples
   - Automation capabilities
   - Benefits and costs

6. **Production Ready**: Battle-tested
   - 31 automated tests
   - Real data validation
   - Feature summary

7. **Future**: What's next
   - More formats
   - Extended testing
   - Enhanced AI
   - Performance analytics
   - Okta authentication

## Speaker Notes

Each slide has detailed speaker notes including:
- What to say (conversational style)
- Key points to emphasize
- Transition phrases
- Example explanations
- Potential questions and answers

Press **P** to see them while presenting!

## Tips for Presenting

1. **Practice the navigation** - especially the ↓ dive and ↑ return movements
2. **Test the speaker console** before your actual presentation
3. **Read through the speaker notes** - they're conversational and detailed
4. **Have the presenter window on your laptop** and project the main presentation
5. **The presentation is fully offline** - no internet needed once files are in place

## Customization

### Change Colors

Edit `css/presentation-style.css`:
```css
:root {
    --primary-color: #2c3e50;    /* Main text color */
    --accent-color: #3498db;      /* Highlights */
    --success-color: #27ae60;     /* Success indicators */
    /* etc. */
}
```

### Add More Slides

Copy an existing slide's structure and adjust the `data-x`, `data-y`, `data-z` coordinates.

### Modify Timing

In each slide's notes, you'll see timing suggestions like "(2-3 minutes)". Adjust based on your presentation length.

## Browser Compatibility

Best viewed in:
- ✅ Chrome/Chromium (recommended)
- ✅ Firefox
- ✅ Safari
- ⚠️  Edge (should work, test beforehand)

## Troubleshooting

**Slides not showing up?**
- Check browser console for errors (F12)
- Make sure all CSS/JS files are in correct folders

**Speaker notes not opening?**
- Allow pop-ups for the presentation page
- Try in a different browser

**Navigation not working?**
- Make sure presentation has focus (click on it)
- Try refreshing the page

**Videos not playing?**
- Check video format compatibility
- Try converting to MP4 (most compatible)

## Credits

- Built with [impress.js](https://impress.js.org/)
- Custom speaker console implementation
- TensorSwitch content and speaker notes by you!

---

**Ready to present?** Open `Diyi_tensorswitch_presentation.html` and press **P** for presenter mode!
