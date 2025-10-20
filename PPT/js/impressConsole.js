/**
 * Simple Impress.js Speaker Console
 * Shows current slide, next slide, and speaker notes
 */

(function(document, window) {
    'use strict';

    var consoleWindow = null;
    var currentStep = null;

    var impressConsole = function() {
        return {
            init: function() {
                // Add event listener for 'P' key to open console
                document.addEventListener('keydown', function(event) {
                    if (event.key === 'p' || event.key === 'P') {
                        impressConsole().open();
                    }
                });

                // Listen for impress:stepenter events
                document.addEventListener('impress:stepenter', function(event) {
                    currentStep = event.target;
                    impressConsole().update();
                });
            },

            open: function() {
                if (consoleWindow && !consoleWindow.closed) {
                    consoleWindow.focus();
                    return;
                }

                consoleWindow = window.open('', 'impressConsole', 'width=1200,height=800,left=0,top=0');

                if (!consoleWindow) {
                    alert('Please allow pop-ups for speaker notes');
                    return;
                }

                // Build console HTML
                var html = `
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>TensorSwitch Presentation - Speaker Notes</title>
                        <link rel="stylesheet" href="css/impressConsole.css">
                        <style>
                            body {
                                margin: 0;
                                padding: 20px;
                                font-family: Arial, sans-serif;
                                background: #f0f0f0;
                            }
                            .console-container {
                                display: grid;
                                grid-template-columns: 40% 60%;
                                grid-template-rows: auto 1fr 1fr;
                                gap: 20px;
                                height: 100vh;
                                max-height: 100vh;
                                padding: 10px;
                            }
                            .timer {
                                grid-column: 1 / -1;
                                background: #2c3e50;
                                color: white;
                                padding: 15px;
                                border-radius: 8px;
                                text-align: center;
                                font-size: 24px;
                                font-weight: bold;
                                display: flex;
                                justify-content: space-between;
                                align-items: center;
                            }
                            .timer-text {
                                flex: 1;
                                text-align: center;
                            }
                            .nav-buttons {
                                display: flex;
                                gap: 10px;
                            }
                            .nav-button {
                                background: #3498db;
                                color: white;
                                border: none;
                                padding: 10px 20px;
                                border-radius: 5px;
                                font-size: 16px;
                                cursor: pointer;
                                font-weight: bold;
                                transition: background 0.3s;
                            }
                            .nav-button:hover {
                                background: #2980b9;
                            }
                            .nav-button:active {
                                background: #1f6391;
                            }
                            .nav-button:disabled {
                                background: #95a5a6;
                                cursor: not-allowed;
                            }
                            .preview-panel {
                                background: white;
                                padding: 15px;
                                border-radius: 8px;
                                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                                overflow: auto;
                            }
                            .preview-panel h3 {
                                margin-top: 0;
                                color: #2c3e50;
                                border-bottom: 2px solid #3498db;
                                padding-bottom: 10px;
                            }
                            .current-slide {
                                border: 3px solid #27ae60;
                            }
                            .next-slide {
                                border: 3px solid #3498db;
                            }
                            .notes-panel {
                                grid-column: 2;
                                grid-row: 2 / 4;
                                background: white;
                                padding: 20px;
                                border-radius: 8px;
                                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                                overflow-y: auto;
                                overflow-x: hidden;
                            }
                            .notes-panel h3 {
                                margin-top: 0;
                                color: #2c3e50;
                                border-bottom: 2px solid #e74c3c;
                                padding-bottom: 10px;
                            }
                            .slide-preview {
                                transform: scale(0.3);
                                transform-origin: top left;
                                width: 333%;
                                height: 333%;
                                pointer-events: none;
                                font-size: 40px;
                            }
                            #notes-content {
                                line-height: 1.8;
                                font-size: 22px;
                            }
                            #notes-content h3 {
                                color: #e74c3c;
                                margin-top: 20px;
                                font-size: 26px;
                            }
                            #notes-content p {
                                margin: 12px 0;
                                font-size: 22px;
                            }
                            #notes-content strong {
                                color: #2c3e50;
                                font-size: 22px;
                            }
                            #notes-content ul {
                                margin: 12px 0;
                                padding-left: 20px;
                                font-size: 22px;
                            }
                            #notes-content li {
                                margin: 8px 0;
                            }
                        </style>
                    </head>
                    <body>
                        <div class="console-container">
                            <div class="timer">
                                <div class="nav-buttons">
                                    <button class="nav-button" id="prev-button" onclick="navigatePrev()">◄ Previous</button>
                                </div>
                                <div class="timer-text" id="timer">00:00:00</div>
                                <div class="nav-buttons">
                                    <button class="nav-button" id="next-button" onclick="navigateNext()">Next ►</button>
                                </div>
                            </div>
                            <div class="preview-panel current-slide">
                                <h3>Current Slide</h3>
                                <div id="current-preview" class="slide-preview"></div>
                            </div>
                            <div class="preview-panel next-slide">
                                <h3>Next Slide</h3>
                                <div id="next-preview" class="slide-preview"></div>
                            </div>
                            <div class="notes-panel">
                                <h3>Speaker Notes</h3>
                                <div id="notes-content"></div>
                            </div>
                        </div>
                        <script>
                            // Timer
                            var startTime = new Date();
                            setInterval(function() {
                                var elapsed = new Date() - startTime;
                                var hours = Math.floor(elapsed / 3600000);
                                var minutes = Math.floor((elapsed % 3600000) / 60000);
                                var seconds = Math.floor((elapsed % 60000) / 1000);
                                document.getElementById('timer').textContent =
                                    pad(hours) + ':' + pad(minutes) + ':' + pad(seconds);
                            }, 1000);

                            function pad(num) {
                                return num < 10 ? '0' + num : num;
                            }

                            // Navigation functions
                            function navigateNext() {
                                if (window.opener && !window.opener.closed) {
                                    window.opener.impress().next();
                                }
                            }

                            function navigatePrev() {
                                if (window.opener && !window.opener.closed) {
                                    window.opener.impress().prev();
                                }
                            }

                            // Keyboard navigation
                            document.addEventListener('keydown', function(event) {
                                if (!window.opener || window.opener.closed) return;

                                switch(event.key) {
                                    case 'ArrowRight':
                                    case 'ArrowDown':
                                    case ' ':
                                    case 'PageDown':
                                        event.preventDefault();
                                        navigateNext();
                                        break;
                                    case 'ArrowLeft':
                                    case 'ArrowUp':
                                    case 'PageUp':
                                        event.preventDefault();
                                        navigatePrev();
                                        break;
                                    case 'Home':
                                        event.preventDefault();
                                        if (window.opener.impress().goto) {
                                            window.opener.impress().goto(0);
                                        }
                                        break;
                                    case 'End':
                                        event.preventDefault();
                                        var steps = window.opener.document.querySelectorAll('.step');
                                        if (steps.length > 0 && window.opener.impress().goto) {
                                            window.opener.impress().goto(steps.length - 1);
                                        }
                                        break;
                                }
                            });
                        </script>
                    </body>
                    </html>
                `;

                consoleWindow.document.write(html);
                consoleWindow.document.close();

                // Wait for console window to load
                setTimeout(function() {
                    impressConsole().update();
                }, 500);
            },

            update: function() {
                if (!consoleWindow || consoleWindow.closed || !currentStep) {
                    return;
                }

                try {
                    // Get current and next slide
                    var steps = Array.from(document.querySelectorAll('.step'));
                    var currentIndex = steps.indexOf(currentStep);
                    var nextStep = steps[currentIndex + 1];

                    // Update current slide preview
                    var currentPreview = consoleWindow.document.getElementById('current-preview');
                    if (currentPreview) {
                        currentPreview.innerHTML = currentStep.innerHTML;
                    }

                    // Update next slide preview
                    var nextPreview = consoleWindow.document.getElementById('next-preview');
                    if (nextPreview) {
                        if (nextStep) {
                            nextPreview.innerHTML = nextStep.innerHTML;
                        } else {
                            nextPreview.innerHTML = '<p style="padding:20px;text-align:center;color:#999;">End of presentation</p>';
                        }
                    }

                    // Update speaker notes
                    var notesContent = consoleWindow.document.getElementById('notes-content');
                    if (notesContent) {
                        var notesDiv = currentStep.querySelector('.notes');
                        if (notesDiv) {
                            notesContent.innerHTML = notesDiv.innerHTML;
                        } else {
                            notesContent.innerHTML = '<p><em>No speaker notes for this slide</em></p>';
                        }
                    }
                } catch (e) {
                    console.error('Error updating console:', e);
                }
            }
        };
    };

    window.impressConsole = impressConsole;

})(document, window);
