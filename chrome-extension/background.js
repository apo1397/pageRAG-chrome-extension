chrome.runtime.onInstalled.addListener(() => {
  console.log('RAG Chrome Extension installed.');
});

chrome.action.onClicked.addListener((tab) => {
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    files: ['content.js']
  });
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "scrapePage") {
    // Forward the message to the popup script
    chrome.runtime.sendMessage(request, (response) => {
      sendResponse(response); // Relay the response back to content.js
    });
    return true; // Indicates that sendResponse will be called asynchronously
  }
});