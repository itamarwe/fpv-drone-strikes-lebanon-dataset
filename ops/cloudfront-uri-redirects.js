// CloudFront viewer-request function. Retire after 2026-10-15 once old links
// have had a three-month migration window.
function handler(event) {
  var request = event.request;
  var oldUri = request.uri;
  var newUri = oldUri;

  if (newUri.indexOf("/annotations/") === 0) {
    newUri = newUri.replace(
      /^\/annotations\/(\d{4})_(\d{2})_(\d{2})_/,
      "/annotations/$1-$2-$3_"
    );
  }
  newUri = newUri.replace(/barashit/g, "biranit");

  if (newUri === oldUri) return request;
  var query = request.querystring;
  var parts = [];
  for (var key in query) {
    if (!Object.prototype.hasOwnProperty.call(query, key)) continue;
    var item = query[key];
    if (item.multiValue) {
      for (var i = 0; i < item.multiValue.length; i++) {
        parts.push(encodeURIComponent(key) + "=" + encodeURIComponent(item.multiValue[i].value));
      }
    } else {
      parts.push(encodeURIComponent(key) + "=" + encodeURIComponent(item.value));
    }
  }
  var location = newUri + (parts.length ? "?" + parts.join("&") : "");
  return {
    statusCode: 308,
    statusDescription: "Permanent Redirect",
    headers: {
      location: { value: location },
      "cache-control": { value: "public, max-age=86400" }
    }
  };
}
