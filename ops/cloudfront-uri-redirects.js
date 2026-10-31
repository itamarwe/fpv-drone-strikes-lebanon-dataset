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
  var redirects = {"2026-05-26_anti_drone_platform_barashit":"2026-05-26_anti_drone_platform_biranit","2026-05-17_buggy_iskandarounah_naqoura":"2026-05-17_military_vehicle_bayyadah_iskandarounah_mmirleb_16443","2026-05-26_humvee_vehicle_in_wadi_near_dibel":"2026-05-26_humvee_zawtar_al_sharqiyah_riverbed_mmirleb_17068","2026-05-06_strike_on_enemy_position":"2026-05-06_position_point_taybeh_mmirleb_16210","2026-05-09_strike_on_soldiers":"2026-05-09_soldiers_gathering_jal_al_alam_mmirleb_15988","2026-05-09_strike_on_soldiers_near_shlomi":"2026-05-09_soldiers_gathering_shlomi_mmirleb_15970","2026-05-12_strike_on_humvee":"2026-05-12_humvee_naqoura_naqoura_iskandarounah_road_mmirleb_16261"};
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
