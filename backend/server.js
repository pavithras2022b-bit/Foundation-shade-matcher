import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import multer from "multer";
import { spawn } from "child_process";
import admin from "firebase-admin";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config();

// Setup Google AI
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Resolve service account JSON (prefer backend/credentials, fallback to old frontend location)
const credPrimary = path.resolve(__dirname, "credentials", "foundation-shade-matcher-firebase-adminsdk-fbsvc-63d7dc25a7.json");
const credFallback = path.resolve(__dirname, "..", "frontend", "foundation-shade-matcher-firebase-adminsdk-fbsvc-63d7dc25a7.json");
let credPath = fs.existsSync(credPrimary) ? credPrimary : (fs.existsSync(credFallback) ? credFallback : null);
if (!credPath) {
  console.error("Firebase service account JSON not found. Expected at:");
  console.error(" - ", credPrimary);
  console.error(" - ", credFallback);
  throw new Error("Missing Firebase service account JSON. Move the file to backend/credentials.");
}
const serviceAccount = JSON.parse(fs.readFileSync(credPath, "utf8"));

// Firebase init
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});
const db = admin.firestore();

const app = express();
app.use(cors());
app.use(bodyParser.json());

// Ensure uploads directory exists and configure multer with absolute path
const uploadsDir = path.resolve(__dirname, "uploads");
fs.mkdirSync(uploadsDir, { recursive: true });
const upload = multer({ dest: uploadsDir });

// Simple request logger to see what arrives
app.use((req, res, next) => {
  console.log(`[req] ${req.method} ${req.url}`);
  next();
});

// Health route FIRST to remove any doubt
app.get("/health", (req, res) => res.json({ status: "ok" }));

// Serve frontend statically so visiting http://localhost:5000 works
const frontendDir = path.resolve(__dirname, "..", "frontend");
console.log("[startup] Serving static from:", frontendDir);
console.log("[startup] index.html exists:", fs.existsSync(path.join(frontendDir, "index.html")));
console.log("[startup] upload.html exists:", fs.existsSync(path.join(frontendDir, "upload.html")));
app.use(express.static(frontendDir));
app.get("/", (req, res) => {
  res.sendFile(path.join(frontendDir, "index.html"));
});
// Explicit route for upload.html in case static middleware is bypassed by cache/proxies
app.get("/upload.html", (req, res) => {
  res.sendFile(path.join(frontendDir, "upload.html"));
});

// ========================
// UPLOAD ENDPOINT
// ========================
app.post("/upload", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No image uploaded" });

    const imagePath = req.file.path;
    const pyScript = path.resolve(__dirname, "ML", "predict_api.py");
    const modelPath = path.resolve(__dirname, "ML", "foundation_model.pkl");

    // 🔥 Run your Python ML prediction
    const py = spawn("python", [pyScript, imagePath, modelPath], { cwd: __dirname });

    let data = "";
    py.stdout.on("data", (chunk) => (data += chunk.toString()));
    py.stderr.on("data", (err) => console.error("⚠️ Python Error:", err.toString()));

    py.on("close", async (code) => {
      try {
        if (code !== 0) {
          console.error("Python process exited with code:", code);
          return res.status(500).json({ error: "Model prediction failed" });
        }

        const result = JSON.parse(data);
        // Save in Firebase
        await db.collection("uploads").add({
          uploadedAt: new Date().toISOString(),
          result,
        });
        res.json({ recommendations: result });
      } catch (err) {
        console.error("⚠️ JSON parse error:", err);
        res.status(500).json({ error: "Invalid response from ML script" });
      } finally {
        // Clean up uploaded file
        fs.unlink(imagePath, () => {});
      }
    });
  } catch (error) {
    console.error("⚠️ Upload error:", error);
    // attempt cleanup if file exists
    if (req?.file?.path) {
      fs.unlink(req.file.path, () => {});
    }
    res.status(500).json({ error: "Upload failed" });
  }
});

// ========================
// QUIZ ENDPOINT
// ========================
app.post("/quiz", (req, res) => {
  try {
    const { answers } = req.body;
    if (!answers || !Array.isArray(answers)) {
      return res.status(400).json({ error: "Invalid answers format" });
    }

    // --- SIMPLE RULE-BASED LOGIC ---
    const skinType = answers[0];
    const coverage = answers[2];
    const finish = answers[3];
    let undertone = "Neutral";
    if (answers[8] === "Green") undertone = "Warm";
    if (answers[8] === "Blue") undertone = "Cool";

    // Basic recommendation logic
    let recProducts = [];
    if (skinType === "Oily" || finish === "Matte") {
       recProducts.push("https://www.maybelline.com/face-makeup/foundation/fit-me-matte-poreless-liquid-foundation");
       recProducts.push("https://www.fentybeauty.com/pro-filt-r-soft-matte-longwear-foundation/FB30006.html");
    } else if (skinType === "Dry" || finish === "Dewy") {
       recProducts.push("https://www.narscosmetics.com/USA/natural-radiant-longwear-foundation/999NAC0000065.html");
       recProducts.push("https://www.armanibeauty.com/makeup/face-makeup/foundation/luminous-silk-foundation/A041.html");
    } else {
       recProducts.push("https://www.lancome-usa.com/makeup/face-makeup/foundation/teint-idole-ultra-wear-24h-longwear-foundation/1000554.html");
       recProducts.push("https://www.esteelauder.com/product/643/22830/product-catalog/makeup/face/foundation/double-wear/stay-in-place-makeup-spf-10");
    }

    const recommendations = {
      skinTone: "Based on your answers",
      undertone: undertone,
      coverage: coverage,
      finish: finish,
      tips: [
        `For ${skinType.toLowerCase()} skin, prep with a suitable primer.`,
        `Since you prefer ${coverage.toLowerCase()} coverage, apply in thin layers to build it up.`,
        `A ${finish.toLowerCase()} finish is great for your skin type!`
      ],
      links: recProducts
    };

    console.log("[QUIZ] Recommendations generated for:", skinType, undertone);
    res.json({ success: true, recommendations });

  } catch (error) {
    console.error("[QUIZ ERROR]", error);
    res.status(500).json({ error: "Failed to process quiz" });
  }
});

// ========================
// CONTACT ENDPOINT
// ========================
app.post("/contact", async (req, res) => {
  try {
    const { name, email, message } = req.body;

    if (!name || !email || !message) {
      return res.status(400).json({ error: "All fields are required" });
    }

    // Save to Firebase "messages" collection
    await db.collection("messages").add({
      name,
      email,
      message,
      sentAt: new Date().toISOString()
    });

    console.log(`[CONTACT] Message received from ${email}`);
    res.json({ success: true });

  } catch (error) {
    console.error("[CONTACT ERROR]", error);
    res.status(500).json({ error: "Failed to send message" });
  }
});

// 404 logger (must be last)
app.use((req, res) => {
  console.log("[404]", req.method, req.url);
  res.status(404).send("Not Found");
});

app.listen(5000, () => console.log("✅ Server running on http://localhost:5000"));