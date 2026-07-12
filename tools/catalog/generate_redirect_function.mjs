#!/usr/bin/env node

import path from "node:path";
import { fileURLToPath } from "node:url";
import { readJson } from "./catalog_lib.mjs";
import fs from "node:fs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const redirects = readJson(path.join(root, "data/redirects.json")).redirects;
const map = Object.fromEntries(redirects.map((item) => [item.from, item.to]));
const code = `// Generated from data/redirects.json. Do not edit by hand.\nfunction encodeQueryPart(value) {\n  try {\n    return encodeURIComponent(decodeURIComponent(value));\n  } catch (error) {\n    return encodeURIComponent(value);\n  }\n}\nfunction handler(event) {\n  var request = event.request;\n  var oldUri = request.uri;\n  var newUri = oldUri;\n  var redirects = ${JSON.stringify(map)};\n  if (newUri.indexOf("/annotations/") === 0) {\n    newUri = newUri.replace(/^\\/annotations\\/(\\d{4})_(\\d{2})_(\\d{2})_/, "/annotations/$1-$2-$3_");\n  }\n  for (var from in redirects) {\n    if (Object.prototype.hasOwnProperty.call(redirects, from)) {\n      newUri = newUri.split(from).join(redirects[from]);\n    }\n  }\n  if (newUri === oldUri) return request;\n  var query = request.querystring;\n  var parts = [];\n  for (var key in query) {\n    if (!Object.prototype.hasOwnProperty.call(query, key)) continue;\n    var item = query[key];\n    if (item.multiValue) {\n      for (var i = 0; i < item.multiValue.length; i++) {\n        parts.push(encodeQueryPart(key) + "=" + encodeQueryPart(item.multiValue[i].value));\n      }\n    } else {\n      parts.push(encodeQueryPart(key) + "=" + encodeQueryPart(item.value));\n    }\n  }\n  var location = newUri + (parts.length ? "?" + parts.join("&") : "");\n  return {\n    statusCode: 308,\n    statusDescription: "Permanent Redirect",\n    headers: {\n      location: { value: location },\n      "cache-control": { value: "public, max-age=86400" }\n    }\n  };\n}\n`;
fs.writeFileSync(path.join(root, "ops/cloudfront-uri-redirects.js"), code);
console.log(`generated CloudFront function with ${redirects.length} explicit redirects`);
