chrome.runtime.onInstalled.addListener(() => {
  console.log('RAG Chrome Extension installed.');
});



chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('Message received in background.js:', request.action);
  console.log('Sender:', sender);
  console.log('Request:', request);
  if (request.action === "scrapePage") {
    // Forward the message to the popup script
    console.log('Forwarding message to popup.js');
    fetch('http://127.0.0.1:8000/save_page', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        url: request.url,
        content: request.content,
        title: request.title,
        favicon_url: request.favicon_url
      })
    })
    .then(response => response.json())
    .then(data => {
      console.log('Page saved successfully in background.js:', data);
      // Send a message back to popup.js to update its UI
      chrome.runtime.sendMessage({ action: 'scrapeComplete', status: 'success', data: data });
      sendResponse({ status: 'success', data: data });
    })
    .catch(error => {
      console.error('Error saving page in background.js:', error);
      chrome.runtime.sendMessage({ action: 'scrapeComplete', status: 'error', error: error });
      sendResponse({ status: 'error', error: error });
    });
    return true; // Indicates that sendResponse will be called asynchronously
  }
});