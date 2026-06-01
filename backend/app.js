async function sendImageToApi(file) {
  const url  = "http://localhost:5000/predict";
; // change when deployed
  const form = new FormData();
  form.append("image", file);
  // optional: add user id
  // form.append("user_id", "user123");

  const res = await fetch(url, {
    method: "POST",
    body: form
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error("Server error: " + txt);
  }
  const data = await res.json();
  return data; // contains { prediction, image_url }
}
