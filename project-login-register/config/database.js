const mysql = require("mysql");

const db = mysql.createConnection({
  host: "localhost",
  user: "root",
  password: "", // ganti dengan password database Anda
  database: "your_database_name", // ganti dengan nama database Anda
});

db.connect((err) => {
  if (err) {
    console.error("Database connection failed:", err);
    return;
  }
  console.log("Connected to the database.");
});

module.exports = db;
