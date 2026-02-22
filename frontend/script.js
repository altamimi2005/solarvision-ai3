document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadSection = document.getElementById('upload-section');
    const loadingSection = document.getElementById('loading-section');
    const resultsSection = document.getElementById('results-section');
    const resultsTbody = document.getElementById('results-tbody');
    const resultsSummary = document.getElementById('results-summary');
    const btnNewUpload = document.getElementById('btn-new-upload');
    const toast = document.getElementById('toast');

    // Drag and drop handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFilesSelect);

    btnNewUpload.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
    });

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFilesSelect(e) {
        const files = e.target.files;
        handleFiles(files);
        fileInput.value = ""; // clear input
    }

    function handleFiles(files) {
        if (!files || files.length === 0) return;

        // Ensure all are images roughly
        const validFiles = Array.from(files).filter(f => f.type.startsWith('image/'));
        if (validFiles.length === 0) {
            showToast('Please upload valid image files.', 'error');
            return;
        }

        uploadAndProcess(validFiles);
    }

    async function uploadAndProcess(validFiles) {
        showLoading();

        const formData = new FormData();
        validFiles.forEach(f => formData.append('files', f));

        try {
            const resp = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!resp.ok) throw new Error('Network response was not ok');

            const data = await resp.json();
            renderResults(data.results);
            showToast('Analysis complete!', 'success');
        } catch (error) {
            console.error('Error:', error);
            showToast('An error occurred during upload or analysis.', 'error');
            hideLoading();
        }
    }

    function renderResults(results) {
        resultsTbody.innerHTML = '';

        if (!results || results.length === 0) {
            hideLoading();
            showToast('No results returned from the server.', 'error');
            return;
        }

        let damagedCount = 0;
        let validImageCount = 0;

        results.forEach(res => {
            const tr = document.createElement('tr');

            // Image Name Cell
            const nameTd = document.createElement('td');
            nameTd.textContent = res.filename || 'Unknown';
            tr.appendChild(nameTd);

            // Result Cell
            const resTd = document.createElement('td');
            const resBadge = document.createElement('span');
            resBadge.className = 'status-badge';

            // Confidence Cell
            const confTd = document.createElement('td');

            if (!res.is_solar_panel) {
                resBadge.className += ' status-error';
                resBadge.textContent = 'Not a solar panel';
                confTd.textContent = 'N/A';
            } else if (res.error) {
                resBadge.className += ' status-error';
                resBadge.textContent = 'Error';
                confTd.textContent = res.error;
            } else {
                validImageCount++;
                if (res.is_damaged) {
                    damagedCount++;
                    resBadge.className += ' status-damaged';
                } else {
                    resBadge.className += ' status-not-damaged';
                }
                resBadge.textContent = res.result;
                confTd.textContent = `${res.confidence}%`;
            }

            resTd.appendChild(resBadge);
            tr.appendChild(resTd);
            tr.appendChild(confTd);

            resultsTbody.appendChild(tr);
        });

        // Summary calculations
        if (validImageCount > 0) {
            const pct = ((damagedCount / validImageCount) * 100).toFixed(1);
            resultsSummary.textContent = `Batch Summary: ${damagedCount} damaged out of ${validImageCount} valid images (${pct}% damaged).`;
        } else {
            resultsSummary.textContent = "No valid solar panels detected in this batch.";
        }

        hideLoadingAndShowResults();
    }

    function showLoading() {
        uploadSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');
    }

    function hideLoading() {
        loadingSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
    }

    function hideLoadingAndShowResults() {
        loadingSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
    }

    let toastTimeout;
    function showToast(message, type = 'success') {
        toast.textContent = message;
        toast.className = `toast show ${type}`;

        clearTimeout(toastTimeout);
        toastTimeout = setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }

    // Check on load if we have a session to restore
    async function checkSession() {
        try {
            const resp = await fetch('/api/results');
            if (resp.ok) {
                const data = await resp.json();
                if (data.results && data.results.length > 0) {
                    renderResults(data.results);
                }
            }
        } catch (e) {
            console.log('No previous session to load.');
        }
    }

    checkSession();
});
