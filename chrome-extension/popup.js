document.addEventListener('DOMContentLoaded', function() {
  const scrapeButton = document.getElementById('scrapePage');
  const queryButton = document.getElementById('queryButton');
  const queryInput = document.getElementById('queryInput');
  const queryResult = document.getElementById('queryResult');
  const savedPagesList = document.getElementById('savedPagesList');

  // Function to fetch and display saved pages
  async function fetchAndDisplaySavedPages() {
    try {
      const response = await fetch('http://127.0.0.1:8000/get_saved_pages');
      const pages = await response.json();
      savedPagesList.innerHTML = ''; // Clear previous list

      if (pages.length === 0) {
        savedPagesList.innerHTML = '<p>No pages saved yet.</p>';
        return;
      }

        const today = new Date();
        today.setHours(0, 0, 0, 0);

        const yesterday = new Date(today);
        yesterday.setDate(today.getDate() - 1);

        const sevenDaysAgo = new Date(today);
        sevenDaysAgo.setDate(today.getDate() - 7);

        const groupedPages = {
            today: [],
            yesterday: [],
            last7Days: [],
            older: []
        };

        pages.forEach(page => {
            const pageDate = new Date(page.date);
            const pageDateOnly = new Date(pageDate.getFullYear(), pageDate.getMonth(), pageDate.getDate());

            if (pageDateOnly.getTime() === today.getTime()) {
                groupedPages.today.push(page);
            } else if (pageDateOnly.getTime() === yesterday.getTime()) {
                groupedPages.yesterday.push(page);
            } else if (pageDateOnly > sevenDaysAgo) {
                groupedPages.last7Days.push(page);
            } else {
                groupedPages.older.push(page);
            }
        });

        function createSection(title, pagesArray) {
            if (pagesArray.length === 0) return;

            const sectionTitle = document.createElement('h3');
            sectionTitle.classList.add('section-subtitle');
            sectionTitle.textContent = title;
            savedPagesList.appendChild(sectionTitle);

            const pagesContainer = document.createElement('div');
            pagesContainer.classList.add('pages-container');

            pagesArray.forEach((page, index) => {
                const pageElement = document.createElement('div');
                pageElement.classList.add('saved-page-item');
                pageElement.dataset.url = page.url; // Store URL for deletion

                if (index >= 3) {
                     pageElement.classList.add('hidden'); // Hide pages beyond the first 3
                 }

                const favicon = document.createElement('img');
                favicon.classList.add('favicon');
                favicon.src = page.favicon_url || 'images/default-favicon.png';
                pageElement.appendChild(favicon);

                const pageContent = document.createElement('div');
                pageContent.classList.add('saved-page-content');

                const titleElement = document.createElement('div');
                titleElement.classList.add('saved-page-title');
                titleElement.textContent = page.title;
                pageContent.appendChild(titleElement);

                const urlElement = document.createElement('a');
                urlElement.classList.add('saved-page-url');
                urlElement.href = page.url;
                // urlElement.textContent = page.url;
                urlElement.target = '_blank'; // Open in new tab
                pageContent.appendChild(urlElement);



                const dateElement = document.createElement('div');
                dateElement.classList.add('saved-page-date');
                const pageDate = new Date(page.date);
                const options = { year: 'numeric', month: 'long', day: 'numeric' };
                dateElement.textContent = pageDate.toLocaleDateString(undefined, options);
                pageContent.appendChild(dateElement);

                pageElement.appendChild(pageContent);

                const deleteButton = document.createElement('button');
                deleteButton.classList.add('button', 'delete-button');
                deleteButton.innerHTML = '<i class="fas fa-trash-alt"></i>'; // FontAwesome trash icon
                deleteButton.addEventListener('click', (event) => {
                    event.stopPropagation(); // Prevent opening the page when deleting
                    if (confirm('Are you sure you want to delete this page?')) {
                        deletePage(page.url, pageElement);
                    }
                });
                pageElement.appendChild(deleteButton);

                pageElement.addEventListener('click', () => {
                    chrome.tabs.create({ url: page.url });
                });

                pagesContainer.appendChild(pageElement);
            });

            savedPagesList.appendChild(pagesContainer);

            if (pagesArray.length > 3) {
                const showMoreButton = document.createElement('button');
                showMoreButton.classList.add('button', 'show-more-button');
                showMoreButton.textContent = `Show All (${pagesArray.length})`;
                showMoreButton.addEventListener('click', () => {
                    const hiddenPages = pagesContainer.querySelectorAll('.saved-page-item.hidden');
                    hiddenPages.forEach(page => {
                        page.classList.remove('hidden');
                    });
                    showMoreButton.style.display = 'none'; // Hide the button after expansion
                });
                savedPagesList.appendChild(showMoreButton);
            }
        }

        createSection('Today', groupedPages.today);
        createSection('Yesterday', groupedPages.yesterday);
        createSection('Last 7 Days', groupedPages.last7Days);
        createSection('Older', groupedPages.older);
    } catch (error) {
      console.error('Error fetching saved pages:', error);
      savedPagesList.innerHTML = '<p>Error loading saved pages.</p>';
    }
  }

  // Function to query pages
  async function queryPages(question) {
    queryResult.textContent = 'Searching...';
    queryResult.classList.remove('hidden');
    try {
      const response = await fetch('http://127.0.0.1:8000/query_pages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question: question })
      });
      const result = await response.json();

      if (result.answer) {
        queryResult.innerHTML = `<b>Answer:</b> ${result.answer}<br>`;
        if (result.source_urls && result.source_urls.length > 0) {
          const sourceUrlsDiv = document.createElement('div');
          sourceUrlsDiv.innerHTML = '<b>Source URLs:</b><br>';
          result.source_urls.forEach(source => {
            const sourceLink = document.createElement('a');
            sourceLink.href = source.url;
            sourceLink.target = '_blank';
            sourceLink.textContent = source.url;
            if (source.favicon_url) {
              const faviconImg = document.createElement('img');
              faviconImg.src = source.favicon_url;
              faviconImg.style.width = '16px';
              faviconImg.style.height = '16px';
              faviconImg.style.marginRight = '5px';
              faviconImg.style.verticalAlign = 'middle';
              sourceUrlsDiv.appendChild(faviconImg);
            }
            sourceUrlsDiv.appendChild(sourceLink);
            sourceUrlsDiv.appendChild(document.createElement('br'));
          });
          queryResult.appendChild(sourceUrlsDiv);
        }
      } else if (result.error) {
        queryResult.textContent = `Error: ${result.error}`;
      } else {
        queryResult.textContent = 'No answer found.';
      }
    } catch (error) {
      console.error('Error querying pages:', error);
      queryResult.textContent = 'Error querying pages.';
    }
  }

  // Event Listeners
  scrapeButton.addEventListener('click', function() {
    scrapeButton.disabled = true;
    scrapeButton.textContent = 'Scraping...';
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.scripting.executeScript({
        target: { tabId: tabs[0].id },
        files: ['content.js']
      }, () => {
        console.log('Content script executed.');
      });
    });
  });

  // Listen for messages from content.js (forwarded by background.js)
  // Function to delete a saved page
  async function deletePage(url, pageElement) {
    try {
      const response = await fetch('http://127.0.0.1:8000/delete_page', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ url: url })
      });

      if (response.ok) {
        pageElement.remove(); // Remove the element from the UI
        // Optionally, re-fetch and display saved pages to update grouping if a section becomes empty
        fetchAndDisplaySavedPages();
      } else {
        console.error('Failed to delete page:', response.statusText);
        alert('Failed to delete page.');
      }
    } catch (error) {
      console.error('Error deleting page:', error);
      alert('Error deleting page.');
    }
  }

  // Event Listeners
  scrapeButton.addEventListener('click', function() {
    scrapeButton.disabled = true;
    scrapeButton.textContent = 'Scraping...';
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.scripting.executeScript({
        target: { tabId: tabs[0].id },
        files: ['content.js']
      }, () => {
        console.log('Content script executed.');
      });
    });
  });

  // Listen for messages from content.js (forwarded by background.js)
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "scrapePage") {
      console.log("Scraping page:", request.url);
      console.log("Scraped content:", request.content);
      console.log("Scraped title:", request.title);
      console.log("Scraped favicon URL:", request.favicon_url);

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
        console.log('Page saved successfully:', data);
        scrapeButton.disabled = false;
        scrapeButton.textContent = 'Scrape Page';
        fetchAndDisplaySavedPages(); // Refresh the list of saved pages
        sendResponse({ status: "Page saved successfully!", data: data });
      })
      .catch(error => {
        console.error('Error saving page:', error);
        scrapeButton.disabled = false;
        scrapeButton.textContent = 'Scrape Page';
        sendResponse({ status: "Error saving page.", error: error });
      });
      return true; // Indicates that sendResponse will be called asynchronously
    }
  });

      queryButton.addEventListener('click', async () => {
        const query = queryInput.value;
        if (!query) return;

        queryResult.textContent = 'Searching...';
        try {
          const response = await fetch('http://127.0.0.1:8000/query_pages', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: query })
          });
          const data = await response.json();
          // Improve text formatting for readability
          let formattedResponse = data.response || 'No answer found.';
          formattedResponse = formattedResponse.replace(/\n/g, '\n\n'); // Add extra line breaks for paragraphs
          formattedResponse = formattedResponse.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>'); // Bold text
          formattedResponse = formattedResponse.replace(/\*(.*?)\*/g, '<em>$1</em>'); // Italic text
          formattedResponse = formattedResponse.replace(/^- (.*)/gm, '• $1'); // List items

          queryResult.innerHTML = formattedResponse;
        } catch (error) {
          console.log('Error querying:', error);
          queryResult.textContent = 'Error querying.';
        }
      });

  // Listen for scrapeComplete message from background.js
  chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.action === 'scrapeComplete') {
      scrapeButton.disabled = false;
      scrapeButton.textContent = 'Scrape Page';
      if (request.status === 'success') {
        console.log('Page scraped successfully!');
        // Refresh the saved pages list after a successful scrape
        fetchAndDisplaySavedPages();
      } else {
        console.error('Failed to scrape page:', request.error);
      }
    }
  });

  // Initial fetch of saved pages when popup opens
  fetchAndDisplaySavedPages();
});