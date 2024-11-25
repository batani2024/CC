const express = require("express");
const bcrypt = require("bcryptjs");
const db = require("../config/database");

const router = express.Router();

// Registrasi
router.post("/register", (req, res) => {
  const { username, email, password } = req.body;

  const hashedPassword = bcrypt.hashSync(password, 10);

  const sql = "INSERT INTO users (username, email, password) VALUES (?, ?, ?)";
  db.query(sql, [username, email, hashedPassword], (err, results) => {
    if (err) {
      return res.status(500).send("Database error");
    }
    res.status(201).send("User registered successfully");
  });
});

// Login
router.post("/login", (req, res) => {
  const { email, password } = req.body;

  const sql = "SELECT * FROM users WHERE email = ?";
  db.query(sql, [email], (err, results) => {
    if (err || results.length === 0) {
      return res.status(400).send("Invalid email or password");
    }

    const user = results[0];
    const isPasswordValid = bcrypt.compareSync(password, user.password);

    if (!isPasswordValid) {
      return res.status(400).send("Invalid email or password");
    }

    req.session.userId = user.id; // Mengatur sesi user
    res.send("Login successful");
  });
});

module.exports = router;
