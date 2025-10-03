// Dataseter Web GUI JavaScript

// Use relative URL to work with any host
const API_URL = window.location.origin;

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
    const depthInput = document.getElementById('web-depth');
    let url = urlInput.value.trim();
    const depth = parseInt(depthInput.value) || 2;

    if (!url) {
        alert('Please enter a valid URL');
        return;
    }

    // Add https:// if no protocol specified
    if (!url.match(/^https?:\/\//i)) {
        url = 'https://' + url;
    }

    addSource({
        type: 'web',
        url: url,
        max_depth: depth
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

    // Check if element exists before updating
    if (!sourceList) {
        console.warn('source-list element not found');
        return;
    }

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
                    ${source.type === 'web' ? `<span class="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded ml-2">Depth: ${source.max_depth}</span>` : ''}
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
        output_format: document.getElementById('output-format').value,
        // Advanced AI features
        chunking_strategy: document.getElementById('chunking-strategy').value,
        extract_knowledge: document.getElementById('extract-knowledge').checked,
        add_metacognitive_annotations: document.getElementById('metacognitive-annotations').checked,
        enable_adversarial_testing: document.getElementById('adversarial-testing').checked
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
                    <button onclick="previewDataset('${job.job_id}')" class="bg-blue-500 text-white px-3 py-1 rounded text-sm hover:bg-blue-600">
                        <i class="fas fa-eye mr-1"></i> Preview & Quality Check
                    </button>
                    ${job.result && job.result.document_count > 0 ? `
                        <button onclick="window.open('${API_URL}/download/${job.job_id}')" class="bg-green-500 text-white px-3 py-1 rounded text-sm hover:bg-green-600 ml-2">
                            Download
                        </button>
                    ` : ''}
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

// Dataset Preview and Quality Check
async function previewDataset(jobId) {
    try {
        const response = await fetch(`${API_URL}/preview/${jobId}?limit=10`);
        if (response.ok) {
            const data = await response.json();
            showPreviewModal(data);
        }
    } catch (error) {
        console.error('Failed to preview dataset:', error);
        alert('Failed to preview dataset');
    }
}

function showPreviewModal(data) {
    // Create modal if it doesn't exist
    let modal = document.getElementById('preview-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'preview-modal';
        modal.className = 'fixed inset-0 bg-black bg-opacity-50 hidden flex items-center justify-center z-50';
        document.body.appendChild(modal);
    }

    // Build quality distribution chart
    const qualityDist = data.quality_metrics.quality_distribution;
    const total = qualityDist.excellent + qualityDist.good + qualityDist.fair + qualityDist.poor;

    modal.innerHTML = `
        <div class="bg-white rounded-lg p-6 max-w-6xl w-full max-h-screen overflow-y-auto m-4">
            <div class="flex justify-between items-start mb-4">
                <h2 class="text-2xl font-bold">Dataset Preview & Quality Check</h2>
                <button onclick="document.getElementById('preview-modal').classList.add('hidden')" class="text-gray-500 hover:text-gray-700">
                    <i class="fas fa-times text-xl"></i>
                </button>
            </div>

            <!-- Quality Metrics -->
            <div class="mb-6">
                <h3 class="text-lg font-semibold mb-3">Quality Metrics</h3>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div class="bg-blue-50 p-3 rounded">
                        <div class="text-sm text-gray-600">Avg Quality Score</div>
                        <div class="text-2xl font-bold ${data.quality_metrics.average_quality_score >= 0.7 ? 'text-green-600' : 'text-orange-600'}">
                            ${data.quality_metrics.average_quality_score}
                        </div>
                    </div>
                    <div class="bg-green-50 p-3 rounded">
                        <div class="text-sm text-gray-600">Avg Text Length</div>
                        <div class="text-2xl font-bold">${Math.round(data.quality_metrics.average_text_length)}</div>
                    </div>
                    <div class="bg-yellow-50 p-3 rounded">
                        <div class="text-sm text-gray-600">Min/Max Length</div>
                        <div class="text-lg font-bold">${data.quality_metrics.min_text_length} / ${data.quality_metrics.max_text_length}</div>
                    </div>
                    <div class="bg-purple-50 p-3 rounded">
                        <div class="text-sm text-gray-600">Total Samples</div>
                        <div class="text-2xl font-bold">${data.total_samples}</div>
                    </div>
                </div>
            </div>

            <!-- Quality Distribution -->
            <div class="mb-6">
                <h3 class="text-lg font-semibold mb-3">Quality Distribution</h3>
                <div class="bg-gray-100 p-4 rounded">
                    <div class="flex items-center space-x-2 mb-2">
                        <span class="text-sm w-20">Excellent</span>
                        <div class="flex-1 bg-gray-300 rounded-full h-6">
                            <div class="bg-green-500 h-full rounded-full flex items-center justify-center text-white text-xs"
                                 style="width: ${(qualityDist.excellent / total * 100) || 0}%">
                                ${qualityDist.excellent}
                            </div>
                        </div>
                    </div>
                    <div class="flex items-center space-x-2 mb-2">
                        <span class="text-sm w-20">Good</span>
                        <div class="flex-1 bg-gray-300 rounded-full h-6">
                            <div class="bg-blue-500 h-full rounded-full flex items-center justify-center text-white text-xs"
                                 style="width: ${(qualityDist.good / total * 100) || 0}%">
                                ${qualityDist.good}
                            </div>
                        </div>
                    </div>
                    <div class="flex items-center space-x-2 mb-2">
                        <span class="text-sm w-20">Fair</span>
                        <div class="flex-1 bg-gray-300 rounded-full h-6">
                            <div class="bg-yellow-500 h-full rounded-full flex items-center justify-center text-white text-xs"
                                 style="width: ${(qualityDist.fair / total * 100) || 0}%">
                                ${qualityDist.fair}
                            </div>
                        </div>
                    </div>
                    <div class="flex items-center space-x-2">
                        <span class="text-sm w-20">Poor</span>
                        <div class="flex-1 bg-gray-300 rounded-full h-6">
                            <div class="bg-red-500 h-full rounded-full flex items-center justify-center text-white text-xs"
                                 style="width: ${(qualityDist.poor / total * 100) || 0}%">
                                ${qualityDist.poor}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Recommendations -->
            <div class="mb-6">
                <h3 class="text-lg font-semibold mb-3">Recommendations</h3>
                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-4">
                    <ul class="list-disc list-inside space-y-1">
                        ${data.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                    </ul>
                </div>
            </div>

            <!-- Sample Preview -->
            <div class="mb-6">
                <h3 class="text-lg font-semibold mb-3">Sample Data Preview</h3>
                <div class="space-y-3">
                    ${data.samples.map(sample => `
                        <div class="border rounded p-3 ${sample.quality_score >= 0.7 ? 'border-green-300' : 'border-orange-300'}">
                            <div class="flex justify-between items-start mb-2">
                                <div class="flex items-center space-x-3">
                                    <span class="text-sm font-semibold">ID: ${sample.id}</span>
                                    <span class="px-2 py-1 rounded text-xs ${sample.quality_score >= 0.7 ? 'bg-green-100 text-green-800' : 'bg-orange-100 text-orange-800'}">
                                        Quality: ${sample.quality_score}
                                    </span>
                                    <span class="text-sm text-gray-600">Length: ${sample.text_length}</span>
                                </div>
                                <span class="text-sm ${sample.has_labels ? 'text-green-600' : 'text-red-600'}">
                                    ${sample.has_labels ? '✓ Has Labels' : '✗ No Labels'}
                                </span>
                            </div>
                            <div class="bg-gray-50 p-2 rounded text-sm font-mono overflow-x-auto">
                                ${sample.text_preview}
                            </div>
                            ${sample.has_labels ? `
                                <div class="mt-2 text-sm">
                                    <span class="font-semibold">Labels:</span> ${sample.labels}
                                </div>
                            ` : ''}
                        </div>
                    `).join('')}
                </div>
            </div>

            <!-- Actions -->
            <div class="flex justify-end gap-3">
                <button onclick="document.getElementById('preview-modal').classList.add('hidden')"
                        class="px-4 py-2 border border-gray-300 rounded hover:bg-gray-100">
                    Close
                </button>
                ${data.total_samples > 0 ? `
                    <button onclick="window.open('${API_URL}/download/${data.job_id}')"
                            class="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600">
                        <i class="fas fa-download mr-2"></i>Download Dataset
                    </button>
                ` : `
                    <span class="text-gray-500 italic">No data available for download</span>
                `}
            </div>
        </div>
    `;

    modal.classList.remove('hidden');
}

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