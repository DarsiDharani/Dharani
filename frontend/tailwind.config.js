// tailwind.config.js
module.exports = {
  content: [
    "./src/**/*.{html,ts}", 
    "./src/app/home/home.component.html",
    "./src/app/engineer-dashboard/engineer-dashboard.component.html", // <--- ADD THIS LINE
  ],
  theme: {
    extend: {
      // Your custom Tailwind themes here
    },
  },
  plugins: [],
}
