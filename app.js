// app.js
const express = require("express");
const bodyParser = require("body-parser");
const { spawn } = require("child_process");

const app = express();
const PORT = 3000;

// Serve static files from the "public" directory
app.use(express.static("public"));
app.use(bodyParser.json());

// Global conversation memory: array of {role: 'user'|'bot', content: string}
let conversationHistory = [];

/**
 * Helper function: calls Python script, returns a Promise of the bot response.
 */
function getBotResponse(conversation) {
  return new Promise((resolve, reject) => {
    let pythonOutput = "";

    // Spawn the python process
    const py = spawn("python", ["serenica_model.py"]);

    // Send JSON conversation to python via STDIN
    py.stdin.write(JSON.stringify(conversation));
    py.stdin.end();

    // Collect output
    py.stdout.on("data", (data) => {
      pythonOutput += data.toString();
    });

    py.stderr.on("data", (err) => {
      console.error("Python error:", err.toString());
    });

    py.on("close", (code) => {
      resolve(pythonOutput.trim());
    });
  });
}

/**
 * POST /api/message
 * Receives { user_input: string }, updates conversation, calls python, returns response
 */
app.post("/api/message", async (req, res) => {
  const userMessage = req.body.user_input;

  // 1) Push user message to conversation memory
  conversationHistory.push({ role: "user", content: userMessage });

  // 2) Call Python to get next bot message
  try {
    const botReply = await getBotResponse(conversationHistory);
    // 3) Add bot response to conversation memory
    conversationHistory.push({ role: "bot", content: botReply });

    // 4) Return bot response as JSON
    res.json({ response: botReply });
  } catch (err) {
    console.error("Error getting bot response:", err);
    res.status(500).json({ error: "Something went wrong." });
  }
});

/**
 * Optional: endpoint to reset conversation
 */
app.post("/api/reset", (req, res) => {
  conversationHistory = [];
  res.json({ status: "conversation reset" });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});
