#include "link.h"
// 0 = normal: binary JPEG stream for tools/save_jpegs.py.
// 1 = debugging: readable status text instead of images (view with `pio device monitor -e bridge`).
#define WAKER_DIAGNOSTICS 0
WiFiClient camera;
mbedtls_gcm_context gcm;
uint8_t *ciphertext=nullptr,*plain=nullptr;
uint8_t previousNonce[8];uint32_t lastSequence=0;bool seen=false;
uint8_t challenge[CHALLENGE];
uint32_t reportAt=0,frameCount=0;
void setup(){
 Serial.begin(USB_BAUD);
 delay(500);
#if WAKER_DIAGNOSTICS
 Serial.printf("\n[Waker] Starting bridge. Free heap: %u\n",ESP.getFreeHeap());
#endif
 ciphertext=(uint8_t*)malloc(MAX_JPEG);plain=(uint8_t*)malloc(MAX_JPEG);
 if(!ciphertext||!plain){Serial.println("[Waker] ERROR: frame buffer allocation failed");while(true)delay(1000);}
 mbedtls_gcm_init(&gcm);
 if(mbedtls_gcm_setkey(&gcm,MBEDTLS_CIPHER_ID_AES,FRAME_KEY,256)){
 Serial.println("[Waker] ERROR: AES key setup failed");while(true)delay(1000);}
 WiFi.mode(WIFI_STA);WiFi.setAutoReconnect(true);
#if WAKER_DIAGNOSTICS
 Serial.printf("[Waker] Connecting to %s. Free heap: %u\n",CAM_SSID,ESP.getFreeHeap());
#endif
 WiFi.begin(CAM_SSID,CAM_PASSWORD);
}
void loop(){
 if(WiFi.status()!=WL_CONNECTED){
  camera.stop();
#if WAKER_DIAGNOSTICS
  if(millis()-reportAt>5000){reportAt=millis();Serial.printf("[Waker] Waiting for Wi-Fi; status=%d, heap=%u\n",(int)WiFi.status(),ESP.getFreeHeap());}
#endif
  delay(500);return;
 }
 if(!camera.connected()){
  camera.stop();
#if WAKER_DIAGNOSTICS
  Serial.printf("[Waker] Wi-Fi connected: IP=%s RSSI=%d; connecting to camera TCP 7443\n",WiFi.localIP().toString().c_str(),WiFi.RSSI());
#endif
  if(!camera.connect(IPAddress(192,168,50,1),7443)){
#if WAKER_DIAGNOSTICS
   Serial.println("[Waker] ERROR: camera TCP connection failed");
#endif
   delay(1000);return;
  }
  // New random challenge per connection (Wi-Fi is up, so the RNG is truly random).
  // Frames recorded from any earlier connection will fail authentication.
  esp_fill_random(challenge,CHALLENGE);
  seen=false;
  if(!writeExact(camera,challenge,CHALLENGE)){
   camera.stop();delay(1000);return;
  }
#if WAKER_DIAGNOSTICS
  Serial.println("[Waker] TCP connected; challenge sent; waiting for frame");
#endif
 }
 uint8_t h[HEADER],tag[16];
 if(!readExact(camera,h,HEADER)){
#if WAKER_DIAGNOSTICS
  Serial.println("[Waker] ERROR: frame header timeout/disconnection");
#endif
  camera.stop();return;
 }
 uint32_t n=get32(h+17),seq=get32(h+5);
 if(memcmp(h,"SLPC",4)||h[4]!=PROTOCOL_VERSION||n==0||n>MAX_JPEG||get32(h+29)!=seq){
#if WAKER_DIAGNOSTICS
  Serial.printf("[Waker] ERROR: invalid header; version=%u length=%u limit=%u\n",h[4],n,MAX_JPEG);
#endif
  camera.stop();return;
 }
 if(!readExact(camera,ciphertext,n)||!readExact(camera,tag,16)){
#if WAKER_DIAGNOSTICS
  Serial.println("[Waker] ERROR: JPEG payload timeout/disconnection");
#endif
  camera.stop();return;
 }
 uint8_t aad[HEADER+CHALLENGE];
 memcpy(aad,h,HEADER);memcpy(aad+HEADER,challenge,CHALLENGE);
 int err=mbedtls_gcm_auth_decrypt(&gcm,n,h+21,12,aad,sizeof aad,tag,16,ciphertext,plain);
 if(err){
#if WAKER_DIAGNOSTICS
  Serial.printf("[Waker] ERROR: authentication/decryption failed (%d). Check identical secrets.h on BOTH boards.\n",err);
#endif
  camera.stop();return;
 }
 // Within one connection the camera cannot reboot, so the boot ID must stay
 // the same and the sequence must strictly increase.
 if(seen&&(memcmp(previousNonce,h+21,8)||seq<=lastSequence)){
#if WAKER_DIAGNOSTICS
  Serial.println("[Waker] ERROR: repeated/out-of-order frame");
#endif
  camera.stop();return;
 }
 memcpy(previousNonce,h+21,8);lastSequence=seq;seen=true;
#if WAKER_DIAGNOSTICS
 ++frameCount;
 if(frameCount==1||frameCount%10==0)Serial.printf("[Waker] OK: decrypted frame seq=%u bytes=%u JPEG markers=%s heap=%u\n",seq,n,(n>=4&&plain[0]==0xff&&plain[1]==0xd8&&plain[n-2]==0xff&&plain[n-1]==0xd9)?"valid":"INVALID",ESP.getFreeHeap());
#else
 Serial.write(h,21);Serial.write(plain,n);Serial.flush();
#endif
 memset(plain,0,n);
}
