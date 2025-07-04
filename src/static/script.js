document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM Content Loaded. Script running.');
    const queryInput = document.getElementById('query-input');
    const submitButton = document.getElementById('submit-button');
    const responseContainer = document.getElementById('response-container');
    const statusBar = document.getElementById('status-bar');
    const indexStatusIcon = document.getElementById('index-status-icon');
    const indexStatusText = document.getElementById('index-status-text');
    const switchIndexButton = document.getElementById('switch-index-button');
    const refreshStatusButton = document.getElementById('refresh-status-button');

    const setInitialStatus = () => {
        statusBar.classList.add('status-building');
        indexStatusText.textContent = 'Initializing...';
        switchIndexButton.classList.add('hidden');
    };

    // Function to update index status
    const updateIndexStatus = async () => {
        console.log('Fetching index status...');
        try {
            const response = await fetch('/index-status');
            console.log('Received response from /index-status. OK:', response.ok, 'Status:', response.status);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Parsed JSON data:', data);

            // Remove all status classes first
            statusBar.classList.remove('status-building', 'status-ready', 'status-synced');

            if (data.green_index_building) {
                statusBar.classList.add('status-building');
                indexStatusText.textContent = `Building Index (${data.green_index.toUpperCase()})...`;
                switchIndexButton.classList.add('hidden');
            } else if (data.green_index_ready_for_swap) {
                statusBar.classList.add('status-ready');
                indexStatusText.textContent = `Index ${data.blue_index.toUpperCase()} is Active`;
                switchIndexButton.textContent = `Switch to Index ${data.green_index.toUpperCase()}`;
                switchIndexButton.classList.remove('hidden');
                switchIndexButton.disabled = false;
            } else {
                statusBar.classList.add('status-synced');
                indexStatusText.textContent = `Index ${data.blue_index.toUpperCase()} is Active`;
                switchIndexButton.classList.add('hidden');
                switchIndexButton.disabled = true;
            }
        } catch (error) {
            console.error('Error fetching index status:', error);
            statusBar.classList.add('status-error'); // Add a visual indicator for error
            indexStatusText.textContent = 'Status: Error - Check console';
            switchIndexButton.classList.add('hidden');
            switchIndexButton.disabled = true;
        }
    };

    // Set initial status, then update
    setInitialStatus();
    setTimeout(updateIndexStatus, 500); // Initial fetch

    // Event listener for the refresh button
    refreshStatusButton.addEventListener('click', updateIndexStatus);

    // Event listener for the switch index button
    switchIndexButton.addEventListener('click', async () => {
        console.log('Switch Index button clicked.');
        switchIndexButton.disabled = true; // Disable button to prevent multiple clicks
        switchIndexButton.textContent = 'Switching...';
        try {
            const response = await fetch('/switch-index', {
                method: 'POST',
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to switch index');
            }
            const data = await response.json();
            console.log(data.message);
            // Immediately trigger a status update to reflect the change
            await updateIndexStatus(); 
        } catch (error) {
            console.error('Error switching index:', error);
            alert(`Error: ${error.message}`);
            // Re-enable button on failure to allow retry
            await updateIndexStatus(); // Refresh status to show the correct state
        }
    });

    submitButton.addEventListener('click', async () => {
        const query = queryInput.value.trim();
        if (!query) return;

        // Display user message
        const userMessageDiv = document.createElement('div');
        userMessageDiv.classList.add('message', 'user-message');
        userMessageDiv.textContent = query;
        responseContainer.appendChild(userMessageDiv);

        queryInput.value = ''; // Clear input
        responseContainer.scrollTop = responseContainer.scrollHeight; // Scroll to bottom

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Display AI response
            const aiMessageDiv = document.createElement('div');
            aiMessageDiv.classList.add('message', 'ai-message');
            aiMessageDiv.textContent = data.response; // Assuming the backend sends a 'response' field
            responseContainer.appendChild(aiMessageDiv);

        } catch (error) {
            console.error('Error:', error);
            const errorMessageDiv = document.createElement('div');
            errorMessageDiv.classList.add('message', 'ai-message');
            errorMessageDiv.style.color = 'red';
            errorMessageDiv.textContent = `Error: ${error.message}. Please try again.`;
            responseContainer.appendChild(errorMessageDiv);
        }
        responseContainer.scrollTop = responseContainer.scrollHeight; // Scroll to bottom
    });

    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            submitButton.click();
        }
    });
});