// Generated from data/redirects.json. Do not edit by hand.
function encodeQueryPart(value) {
  try {
    return encodeURIComponent(decodeURIComponent(value));
  } catch (error) {
    return encodeURIComponent(value);
  }
}
function handler(event) {
  var request = event.request;
  var oldUri = request.uri;
  var newUri = oldUri;
  var redirects = {"2026-05-26_anti_drone_platform_barashit":"2026-05-26_anti_drone_platform_biranit"};
  if (newUri.indexOf("/annotations/") === 0) {
    newUri = newUri.replace(/^\/annotations\/(\d{4})_(\d{2})_(\d{2})_/, "/annotations/$1-$2-$3_");
  }
  for (var from in redirects) {
    if (Object.prototype.hasOwnProperty.call(redirects, from)) {
      newUri = newUri.split(from).join(redirects[from]);
    }
  }
  if (newUri === oldUri) return request;
  var query = request.querystring;
  var parts = [];
  for (var key in query) {
    if (!Object.prototype.hasOwnProperty.call(query, key)) continue;
    var item = query[key];
    if (item.multiValue) {
      for (var i = 0; i < item.multiValue.length; i++) {
        parts.push(encodeQueryPart(key) + "=" + encodeQueryPart(item.multiValue[i].value));
      }
    } else {
      parts.push(encodeQueryPart(key) + "=" + encodeQueryPart(item.value));
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
