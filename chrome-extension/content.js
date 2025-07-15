(function() { console.log('Content script loaded - Script started');
  // Ensure the script only runs once per page load
  if (document.body.dataset.hypScrapeProcessed) {
    return;
  }
  document.body.dataset.hypScrapeProcessed = 'true';

  const mainContentSelectors = [
    'article',
    'main',
    '[role="main"]',
    'div.entry-content',
    'div.post-content',
    'div.article-content',
    'div.main-content',
    'div#content',
    'div#main'
  ];

  const generalTextSelectors = [
    'p',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'li',
    'blockquote',
    'pre',
    'code'
  ];

  let content = '';

  // Try to find the main content block first
  for (const selector of mainContentSelectors) {
    const mainElement = document.querySelector(selector);
    if (mainElement) {
      // If a main content element is found, extract text from within it
      // and then clean it up.
      content = mainElement.innerText;
      break; // Found the main content, no need to check other selectors
    }
  }

  // If no main content block was found, fall back to collecting text from general elements
  if (content.trim() === '') {
    let extractedText = [];
    for (const selector of generalTextSelectors) {
      const elements = document.querySelectorAll(selector);
      elements.forEach(element => {
        // Exclude script and style tags
        if (element.tagName !== 'SCRIPT' && element.tagName !== 'STYLE') {
          const text = element.innerText.trim();
          if (text) {
            extractedText.push(text);
          }
        }
      });
    }
    content = extractedText.join('\n\n');
  }

// Join the extracted text and clean up multiple newlines/spaces
  // Final cleaning steps
  content = content.replace(/\n\s*\n/g, '\n\n'); // Replace multiple newlines with two
  content = content.replace(/\s\s+/g, ' '); // Replace multiple spaces with one

  // Fallback to body innerText if still no content
  if (content.trim() === '') {
    content = document.body.innerText;
  }

  const pageUrl = window.location.href;
  const pageTitle = document.title;
  const faviconEl = document.querySelector('link[rel="icon"], link[rel="shortcut icon"]');
  const faviconUrl = faviconEl ? faviconEl.href : null;

  console.log("Content script running on:", pageUrl);

    console.log('Sending message to background.js with action: scrapePage');
    const now = new Date();
    const sevenDaysAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000); // 7 days in milliseconds
    const randomTimestamp = new Date(sevenDaysAgo.getTime() + Math.random() * (now.getTime() - sevenDaysAgo.getTime()));

    chrome.runtime.sendMessage({
      action: "scrapePage",
      url: window.location.href,
      title: pageTitle,
      content: content || '',
      faviconUrl: faviconUrl,
      timestamp: randomTimestamp.toISOString() // Add the random timestamp here
    }, function(response) {
      console.log('Message sent to background.js, response:', response);
    });
})();