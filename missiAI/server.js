const express = require('express');
const path = require('path');
const cors = require('cors');
const { createProxyMiddleware } = require('http-proxy-middleware');
const fs = require('fs');
const app = express();
const port = 8006;

// Enable CORS for all routes
app.use(cors());

// Parse JSON bodies
app.use(express.json());

// Serve static files
app.use(express.static(path.join(__dirname)));

// Create output directory if it doesn't exist
const outputDir = path.join(__dirname, '..', 'output');
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
  console.log('Created output directory:', outputDir);
}

// Serve output directory for audio files
app.use('/output', express.static(outputDir));

// Debug endpoint to list output directory files
app.get('/debug/output-files', (req, res) => {
  try {
    const files = fs.readdirSync(outputDir);
    const fileDetails = files.map(file => {
      const stats = fs.statSync(path.join(outputDir, file));
      return {
        filename: file,
        path: `/output/${file}`,
        size: stats.size,
        created: stats.birthtime
      };
    });
    res.json({
      outputDir,
      files: fileDetails
    });
  } catch (error) {
    res.status(500).json({
      error: error.message,
      outputDir
    });
  }
});

// Proxy API requests to the MiniMax Direct API
app.use('/api', createProxyMiddleware({
  target: 'http://localhost:8005',
  changeOrigin: true,
  pathRewrite: {
    '^/api': ''
  },
  onProxyReq: (proxyReq, req, res) => {
    // Log proxy requests
    console.log('Proxying request to MiniMax Direct API:', req.method, req.path);
  }
}));

// Serve index.html for the root route
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    minimax_api_url: 'http://localhost:8005'
  });
});

// Start the server
app.listen(port, () => {
  console.log(`EmergencyAI server running at http://localhost:${port}`);
  console.log(`Serving audio files from: ${outputDir}`);
  console.log(`Proxying API requests to: http://localhost:8005`);
});
