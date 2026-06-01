import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import { 
  getAuth, 
  createUserWithEmailAndPassword, 
  signInWithEmailAndPassword, 
  signOut, 
  onAuthStateChanged 
} from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";

// Your Config (Same as in your other files)
const firebaseConfig = {
  apiKey: "AIzaSyCLuH8sXI6JaR-seKfSO1PYF1ux6zvD1Hw",
  authDomain: "foundation-finder-3dd16.firebaseapp.com",
  projectId: "foundation-finder-3dd16",
  storageBucket: "foundation-finder-3dd16.firebasestorage.app",
  messagingSenderId: "95783410327",
  appId: "1:95783410327:web:afe1c287dd684a142ce7d2"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// --- DOM Elements ---
const signinForm = document.getElementById('signinForm');
const signupForm = document.getElementById('signupForm');
const authMsg = document.getElementById('auth-msg');
const navMenu = document.querySelector('.nav-menu');

// --- 1. Handle Sign Up ---
if (signupForm) {
  signupForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = document.getElementById('signup-email').value;
    const password = document.getElementById('signup-password').value;
    
    try {
      await createUserWithEmailAndPassword(auth, email, password);
      window.location.href = 'index.html'; // Redirect to home on success
    } catch (error) {
      showError(error.message);
    }
  });
}

// --- 2. Handle Sign In ---
if (signinForm) {
  signinForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = document.getElementById('signin-email').value;
    const password = document.getElementById('signin-password').value;

    try {
      await signInWithEmailAndPassword(auth, email, password);
      window.location.href = 'index.html'; // Redirect home
    } catch (error) {
      showError("Invalid email or password.");
    }
  });
}

// --- 3. Handle Global Auth State (Update Navbar) ---
onAuthStateChanged(auth, (user) => {
  if (user) {
    // If logged in, add a Logout button to the nav
    const logoutLink = document.createElement('a');
    logoutLink.href = "#";
    logoutLink.textContent = "Logout";
    logoutLink.addEventListener('click', () => signOut(auth));
    
    // Remove generic "Login" link if it exists
    const loginLink = document.querySelector('a[href="login.html"]');
    if (loginLink) loginLink.remove();
    
    navMenu.appendChild(logoutLink);
  }
});

// Helper to show errors
function showError(msg) {
  if (authMsg) {
    authMsg.textContent = msg.replace('Firebase: ', '');
  } else {
    alert(msg);
  }
}