// Dataseter Web GUI JavaScript

const API_URL = 'http://localhost:8000';

let sources = [];
let currentJobId = null;

// Page Navigation
function showPage(pageName) {
    document.querySelectorAll('.page').forEach(page => {
        page.classList.add('hidden');
    });
    document.getElementById(`${pageName}-page`).classList.remove('hidden');

    if (pageName === 'jobs') {
        loadJobs();
    }
}

// File Handling
function handleFileSelect(event) {
    const files = event.target.files;
    Array.from(files).forEach(file => {
        uploadFile(file);
    });
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            addSource({
                type: detectFileType(file.name),
                path: data.path,
                filename: data.filename,
                size: data.size
            });
            updateSourceList();
        }
    } catch (error) {
        console.error('Upload failed:', error);
        alert('Failed to upload file');
    }
}

function detectFileType(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const typeMap = {
        'pdf': 'pdf',
        'docx': 'office',
        'doc': 'office',
        'xlsx': 'office',
        'xls': 'office',
        'pptx': 'office',
        'ppt': 'office',
        'epub': 'ebook',
        'mobi': 'ebook',
        'txt': 'text',
        'md': 'text'
    };
    return typeMap[ext] || 'text';
}

function addWebSource() {
    const urlInput = document.getElementById('web-url');
    const url = urlInput.value.trim();

    if (!url) {
        alert('Please enter a valid URL');
        return;
    }

    addSource({
        type: 'web',
        url: url,
        max_depth: 2
    });

    urlInput.value = '';
    updateSourceList();
}

function addSource(source) {
    sources.push(source);
}

function removeSource(index) {
    sources.splice(index, 1);
    updateSourceList();
}

function updateSourceList() {
    const sourceList = document.getElementById('source-list');

    if (sources.length === 0) {
        sourceList.innerHTML = '<p class="text-gray-500">No sources added yet</p>';
        return;
    }

    sourceList.innerHTML = '<h4 class="font-semibold mb-2">Added Sources:</h4>' +
        sources.map((source, index) => `
            <div class="flex items-center justify-between bg-gray-100 p-2 rounded mb-2">
                <div class="flex items-center">
                    <i class="fas ${getSourceIcon(source.type)} mr-2"></i>
                    <span>${source.filename || source.url}</span>
                    <span class="text-sm text-gray-500 ml-2">(${source.type})</span>
                </div>
                <button onclick="removeSource(${index})" class="text-red-500 hover:text-red-700">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `).join('');
}

function getSourceIcon(type) {
    const icons = {
        'pdf': 'fa-file-pdf',
        'office': 'fa-file-word',
        'ebook': 'fa-book',
        'text': 'fa-file-alt',
        'web': 'fa-globe'
    };
    return icons[type] || 'fa-file';
}

// Extraction
async function startExtraction() {
    if (sources.length === 0) {
        alert('Please add at least one source');
        return;
    }

    const request = {
        sources: sources,
        chunk_size: parseInt(document.getElementById('chunk-size').value),
        overlap: parseInt(document.getElementById('overlap').value),
        quality_threshold: parseFloat(document.getElementById('quality-threshold').value),
        remove_pii: document.getElementById('remove-pii').checked,
        output_format: document.getElementById('output-format').value
    };

    try {
        const response = await fetch(`${API_URL}/extract`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(request)
        });

        if (response.ok) {
            const data = await response.json();
            currentJobId = data.job_id;
            showProgressModal();
            monitorJob(currentJobId);
        }
    } catch (error) {
        console.error('Extraction failed:', error);
        alert('Failed to start extraction');
    }
}

// Job Monitoring
async function monitorJob(jobId) {
    const interval = setInterval(async () => {
        try {
            const response = await fetch(`${API_URL}/job/${jobId}`);
            if (response.ok) {
                const job = await response.json();
                updateProgress(job.progress, job.message);

                if (job.status === 'completed') {
                    clearInterval(interval);
                    onJobComplete(job);
                } else if (job.status === 'failed') {
                    clearInterval(interval);
                    onJobFailed(job);
                }
            }
        } catch (error) {
            console.error('Failed to get job status:', error);
        }
    }, 1000);
}

function updateProgress(progress, message) {
    document.getElementById('progress-bar').style.width = `${progress}%`;
    document.getElementById('progress-message').textContent = message;
}

function onJobComplete(job) {
    hideProgressModal();
    alert('Dataset creation completed!');

    // Show download button
    const downloadUrl = `${API_URL}/download/${job.job_id}`;
    if (confirm('Download the dataset?')) {
        window.open(downloadUrl, '_blank');
    }

    // Show statistics
    if (job.result && job.result.statistics) {
        showStatistics(job.result.statistics);
    }

    // Refresh jobs list
    showPage('jobs');
}

function onJobFailed(job) {
    hideProgressModal();
    alert(`Job failed: ${job.message}`);
}

// Jobs Management
async function loadJobs() {
    // In a real implementation, we'd fetch jobs from the server
    // For now, show current job if exists
    if (currentJobId) {
        try {
            const response = await fetch(`${API_URL}/job/${currentJobId}`);
            if (response.ok) {
                const job = await response.json();
                displayJobs([job]);
            }
        } catch (error) {
            console.error('Failed to load jobs:', error);
        }
    } else {
        document.getElementById('jobs-list').innerHTML = '<p class="text-gray-500">No active jobs</p>';
    }
}

function displayJobs(jobs) {
    const jobsList = document.getElementById('jobs-list');
    jobsList.innerHTML = jobs.map(job => `
        <div class="border rounded p-4 mb-4">
            <div class="flex justify-between items-start mb-2">
                <div>
                    <span class="font-semibold">Job ID:</span> ${job.job_id}
                </div>
                <span class="px-2 py-1 rounded text-sm ${getStatusClass(job.status)}">
                    ${job.status}
                </span>
            </div>
            <div class="mb-2">
                <div class="bg-gray-200 rounded-full h-2 overflow-hidden">
                    <div class="bg-blue-600 h-full" style="width: ${job.progress}%"></div>
                </div>
            </div>
            <p class="text-sm text-gray-600">${job.message}</p>
            ${job.status === 'completed' ? `
                <div class="mt-4">
                    <button onclick="window.open('${API_URL}/download/${job.job_id}')" class="bg-green-500 text-white px-3 py-1 rounded text-sm hover:bg-green-600">
                        Download
                    </button>
                    <button onclick="deleteJob('${job.job_id}')" class="bg-red-500 text-white px-3 py-1 rounded text-sm hover:bg-red-600 ml-2">
                        Delete
                    </button>
                </div>
            ` : ''}
        </div>
    `).join('');
}

function getStatusClass(status) {
    const classes = {
        'processing': 'bg-blue-100 text-blue-800',
        'completed': 'bg-green-100 text-green-800',
        'failed': 'bg-red-100 text-red-800'
    };
    return classes[status] || 'bg-gray-100 text-gray-800';
}

async function deleteJob(jobId) {
    if (!confirm('Delete this job and its output?')) {
        return;
    }

    try {
        const response = await fetch(`${API_URL}/job/${jobId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            if (jobId === currentJobId) {
                currentJobId = null;
            }
            loadJobs();
        }
    } catch (error) {
        console.error('Failed to delete job:', error);
    }
}

// UI Helpers
function showProgressModal() {
    document.getElementById('progress-modal').classList.remove('hidden');
}

function hideProgressModal() {
    document.getElementById('progress-modal').classList.add('hidden');
}

function clearAll() {
    sources = [];
    updateSourceList();
    document.getElementById('file-input').value = '';
    document.getElementById('web-url').value = '';
}

function showStatistics(stats) {
    // Display statistics in analyze page
    const analysisContent = document.getElementById('analysis-content');
    analysisContent.innerHTML = `
        <div class="grid grid-cols-2 gap-4">
            <div class="bg-gray-100 p-4 rounded">
                <h4 class="font-semibold">Total Documents</h4>
                <p class="text-2xl">${stats.total_documents || 0}</p>
            </div>
            <div class="bg-gray-100 p-4 rounded">
                <h4 class="font-semibold">Total Words</h4>
                <p class="text-2xl">${stats.total_words || 0}</p>
            </div>
            <div class="bg-gray-100 p-4 rounded">
                <h4 class="font-semibold">Vocabulary Size</h4>
                <p class="text-2xl">${stats.vocabulary_size || 0}</p>
            </div>
            <div class="bg-gray-100 p-4 rounded">
                <h4 class="font-semibold">Quality Score</h4>
                <p class="text-2xl">${stats.quality_stats?.mean.toFixed(2) || 'N/A'}</p>
            </div>
        </div>
    `;
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    updateSourceList();
});

// Drag and Drop
const dropZone = document.querySelector('.border-dashed');

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('bg-gray-100');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('bg-gray-100');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('bg-gray-100');

    const files = e.dataTransfer.files;
    Array.from(files).forEach(file => {
        uploadFile(file);
    });
});