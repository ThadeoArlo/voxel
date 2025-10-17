#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>

// ---- Wi-Fi ----
const char* ssid = "Founders Guest";
const char* password = "localhost0";
// const char* ssid = "Zahav";
// const char* password = "crunchyfalafel";

// ---- Camera Identification ----
const char* cameraName = "ESP32-CAM"; // Change this for each camera: CAM-1, CAM-2, CAM-3

// ---- Static IP Configuration ----
// Configure each camera with unique static IP (matching your network: 10.104.x.x)
IPAddress local_IP(10, 104, 14, 201);  // Change last number for each camera: 201, 202, 203
IPAddress gateway(10, 104, 14, 1);     // Your router's IP (usually .1)
IPAddress subnet(255, 255, 0, 0);      // /16 subnet mask
IPAddress primaryDNS(8, 8, 8, 8);      // Google DNS
IPAddress secondaryDNS(8, 8, 4, 4);    // Google DNS

// ---- Camera pin map for AI-Thinker ESP32-CAM (OV2640) ----
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

WebServer server(80);

void handleRoot() {
  String html = "<html><head><title>" + String(cameraName) + "</title></head><body>";
  html += "<h3>" + String(cameraName) + " Stream</h3>";
  html += "<p>IP: " + WiFi.localIP().toString() + "</p>";
  html += "<p>Go to <a href='/stream'>/stream</a> for live video</p>";
  html += "<p>Go to <a href='/jpg'>/jpg</a> for single frame</p>";
  html += "<p>Go to <a href='/status'>/status</a> for camera info</p>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

void setup() {
  Serial.begin(115200);
  Serial.println("\nBooting...");

  // Camera configuration
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.fb_location = CAMERA_FB_IN_PSRAM;      // keep frame buffers in PSRAM for speed
  config.grab_mode = CAMERA_GRAB_LATEST;        // drop old frames to minimize latency

  if (psramFound()) {
    // Ultra-fast profile: smallest frame, low bitrate, double buffering
    config.frame_size = FRAMESIZE_QQVGA; // 160x120 → maximum FPS
    config.jpeg_quality = 24;            // higher number = lower quality, faster
    config.fb_count = 2;                 // double buffering for throughput; 3 may add latency
  } else {
    config.frame_size = FRAMESIZE_QQVGA; // 160x120
    config.jpeg_quality = 26;
    config.fb_count = 1;
  }

  // Initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    ESP.restart();
  }

  // Optional: basic sensor tuning for speed
  sensor_t * s = esp_camera_sensor_get();
  if (s) {
    s->set_brightness(s, 0);     // -2 to 2
    s->set_contrast(s, 0);       // -2 to 2
    s->set_saturation(s, 0);     // -2 to 2
    s->set_special_effect(s, 0); // 0-6
    s->set_whitebal(s, 1);       // 0 = disable , 1 = enable
    s->set_awb_gain(s, 1);
    s->set_wb_mode(s, 0);        // 0-4
    s->set_exposure_ctrl(s, 1);
    s->set_aec2(s, 0);
    s->set_ae_level(s, 0);       // -2 to 2
    s->set_aec_value(s, 1200);   // 0 to 1200
    s->set_gain_ctrl(s, 1);
    s->set_agc_gain(s, 0);       // 0 to 30
    s->set_gainceiling(s, (gainceiling_t)2);
    s->set_bpc(s, 0);            // black pixel correction
    s->set_wpc(s, 0);            // white pixel correction
    s->set_raw_gma(s, 0);
    s->set_lenc(s, 0);           // lens correction off for speed
    s->set_hmirror(s, 0);
    s->set_vflip(s, 0);
    s->set_dcw(s, 1);            // downsize enable
    s->set_colorbar(s, 0);
  }

  // Configure static IP
  if (!WiFi.config(local_IP, gateway, subnet, primaryDNS, secondaryDNS)) {
    Serial.println("Static IP configuration failed");
  }

  // Connect to Wi-Fi
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false); // disable power save to improve throughput/latency
  WiFi.setTxPower(WIFI_POWER_19_5dBm); // maximize RF power for link quality
  WiFi.begin(ssid, password);
  Serial.printf("Connecting to %s with static IP %s...\n", ssid, local_IP.toString().c_str());
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.println("Static IP: " + WiFi.localIP().toString());

  // Optional mDNS with unique names
  String mdnsName = String(cameraName);
  mdnsName.toLowerCase();
  if (MDNS.begin(mdnsName.c_str())) {
    Serial.println("mDNS responder started: http://" + mdnsName + ".local");
  }

  // Web routes
  server.on("/", HTTP_GET, handleRoot);
  server.on("/jpg", HTTP_GET, []() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) { server.send(500, "text/plain", "Camera error"); return; }
    server.send_P(200, "image/jpeg", (const char*)fb->buf, fb->len);
    esp_camera_fb_return(fb);
  });
  server.on("/status", HTTP_GET, []() {
    String json = "{";
    json += "\"name\":\"" + String(cameraName) + "\",";
    json += "\"ip\":\"" + WiFi.localIP().toString() + "\",";
    json += "\"mac\":\"" + WiFi.macAddress() + "\",";
    json += "\"rssi\":" + String(WiFi.RSSI()) + ",";
    json += "\"uptime\":" + String(millis()) + ",";
    json += "\"freeHeap\":" + String(ESP.getFreeHeap());
    json += "}";
    server.send(200, "application/json", json);
  });
  server.on("/stream", HTTP_GET, []() {
    WiFiClient client = server.client();
    String resp = "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
    client.print(resp);
    while (client.connected()) {
      camera_fb_t *fb = esp_camera_fb_get();
      if (!fb) continue;
      client.printf("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
      client.write(fb->buf, fb->len);
      client.print("\r\n");
      esp_camera_fb_return(fb);
      // No delay for maximum throughput; router/AP will become limiting factor
    }
  });

  server.begin();
  Serial.println("Server ready.");
}

void loop() {
  server.handleClient();
}
