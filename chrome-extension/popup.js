document.addEventListener('DOMContentLoaded', function() {
  console.log('Popup DOMContentLoaded - Script started');
  const scrapeButton = document.getElementById('scrapePage');
  const queryButton = document.getElementById('queryButton');
  const queryInput = document.getElementById('queryInput');
  const queryResult = document.getElementById('queryResult');
  const savedPagesList = document.getElementById('savedPagesList');
  const suggestedQueriesList = document.querySelector('.suggested-queries ul');

  if (suggestedQueriesList) {
    suggestedQueriesList.addEventListener('click', (event) => {
      if (event.target.tagName === 'LI') {
        queryInput.value = event.target.textContent;
      }
    });
  }

  // Function to fetch and display saved pages
  async function fetchAndDisplaySavedPages() {
    try {
      const userTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
      const response = await fetch(`http://127.0.0.1:8000/get_saved_pages?timezone=${encodeURIComponent(userTimezone)}`);
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

                const pageContent = document.createElement('div');
                pageContent.classList.add('saved-page-content');
                pageElement.appendChild(pageContent); // Move this line up

                const favicon = document.createElement('img');
                favicon.classList.add('favicon');
                favicon.style.width = '16px';
                favicon.style.height = '16px';
                favicon.style.marginRight = '5px';
                let faviconUrl = page.favicon_url;
                console.log('Original favicon URL:', faviconUrl);
                
                if (faviconUrl) {
                    // Check if favicon URL is a relative path
                    if (faviconUrl.startsWith('/') || !faviconUrl.startsWith('http')) {
                        // Get domain from page URL
                        const pageUrl = new URL(page.url);
                        const domain = `${pageUrl.protocol}//${pageUrl.host}`;
                        
                        // Handle both absolute and relative paths
                        faviconUrl = faviconUrl.startsWith('/') 
                            ? `${domain}${faviconUrl}`
                            : `${domain}/${faviconUrl}`;
                        console.log('Processed favicon URL:', faviconUrl);
                    }
                    favicon.src = faviconUrl;
                } else {
                    console.log('No favicon URL found, using default');
                    favicon.src = 'images/default-favicon.png';
                }
                pageContent.appendChild(favicon);

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
                // Ensure the date is parsed correctly
                const pageDate = new Date(page.date);
                const options = { year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' };
                dateElement.textContent = pageDate.toLocaleDateString(undefined, options);
                pageContent.appendChild(dateElement);

                // pageElement.appendChild(pageContent); // This line is moved up

                const deleteButton = document.createElement('button');
                deleteButton.classList.add('button', 'delete-button');
                deleteButton.innerHTML = '<i class="fas fa-trash-alt"></i>'; // FontAwesome trash icon
                deleteButton.addEventListener('click', (event) => {
                    console.log('Delete button clicked - Initiating delete');
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
      const userTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
      const response = await fetch('http://127.0.0.1:8000/query_pages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question: question, timezone: userTimezone })
      });
      const result = await response.json();

      if (result.answer) {
        queryResult.innerHTML = `<b>Answer:</b> ${result.answer}<br>`;
        if (result.source_urls && result.source_urls.length > 0) {
          const sourceUrlsDiv = document.createElement('div');
          sourceUrlsDiv.innerHTML = '<b>Source URLs:</b><br>';
          sourceUrlsDiv.style.wordWrap = 'break-word';
          result.source_urls.forEach(source => {
            const sourceLink = document.createElement('a');
            sourceLink.href = source.url;
            sourceLink.target = '_blank';
            sourceLink.textContent = source.title || source.url;
            sourceLink.style.display = 'block';
            sourceLink.style.display = 'block';
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
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('Message received in popup.js:', request.action);
    console.log('Sender:', sender);
    console.log('Request:', request);

    if (request.action === "scrapePage") {
      // The actual saving is now handled by background.js
      // This block in popup.js is no longer responsible for the fetch call.
      // It will receive a 'scrapeComplete' message from background.js instead.
      console.log('popup.js: scrapePage message received, but saving handled by background.js');
      // No need to send a response here, as background.js will send scrapeComplete
    } else if (request.action === 'scrapeComplete') {
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

  queryButton.addEventListener('click', async () => {
    console.log('Query button clicked - Initiating query');
    const query = queryInput.value;
    if (!query) return;

    queryResult.textContent = 'Searching...';
    try {
      const userTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
      const response = await fetch('http://127.0.0.1:8000/query_pages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question: query, timezone: userTimezone })
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
            sourceLink.textContent = source.title || source.url;
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
  });

  // Initial fetch of saved pages when popup opens
  fetchAndDisplaySavedPages();
});