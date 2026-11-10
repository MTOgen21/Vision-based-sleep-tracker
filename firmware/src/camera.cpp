#include "link.h"
#include "esp_camera.h"
#include "esp_timer.h"
#include "esp_system.h"
WiFiServer server(7443);
WiFiClient peer;
mbedtls_gcm_context gcm;
uint8_t* encrypted;
uint8_t bootNonce[8];
uint8_t challenge[CHALLENGE];
uint32_t sequence=0;
constexpr uint32_t TARGET_FPS=6;
constexpr uint32_t FRAME_MS=1000/TARGET_FPS;
// Must stay well under the bridge's 5 s read timeout in link.h.
constexpr uint32_t SETTLE_MS=3000;
uint32_t lockAt=0;bool locked=false;
// Lit-room setting: revisit for IR camera.
void setAuto(bool on){
 sensor_t* s=esp_camera_sensor_get();
 s->set_exposure_ctrl(s,on);s->set_gain_ctrl(s,on);s->set_whitebal(s,on);
}
void setup(){
 Serial.begin(115200);
 if(!psramFound()){Serial.println("PSRAM required");while(true)delay(1000);}
 camera_config_t c={};
 c.ledc_channel=LEDC_CHANNEL_0;c.ledc_timer=LEDC_TIMER_0;
 c.pin_d0=5;c.pin_d1=18;c.pin_d2=19;c.pin_d3=21;c.pin_d4=36;c.pin_d5=39;c.pin_d6=34;c.pin_d7=35;
 c.pin_xclk=0;c.pin_pclk=22;c.pin_vsync=25;c.pin_href=23;
 c.pin_sccb_sda=26;c.pin_sccb_scl=27;c.pin_pwdn=32;c.pin_reset=-1;
 c.xclk_freq_hz=20000000;c.pixel_format=PIXFORMAT_JPEG;c.frame_size=FRAMESIZE_VGA;
 c.jpeg_quality=15;c.fb_count=2;c.grab_mode=CAMERA_GRAB_LATEST;c.fb_location=CAMERA_FB_IN_PSRAM;
 if(esp_camera_init(&c)!=ESP_OK){Serial.println("Camera init failed");while(true)delay(1000);}
 // 50 Hz mains: sensor-bank COM3 (bank 1, address 0x0C), auto banding off, 50 Hz on.
 // The sensor powers up at 60 Hz, which lets room lights flicker into the image.
 sensor_t* s=esp_camera_sensor_get();
 s->set_reg(s,0x10C,0x06,0x04);
 // The camera is mounted upside down: flip and mirror together rotate the image 180°.
 s->set_vflip(s,1);s->set_hmirror(s,1);
 encrypted=(uint8_t*)ps_malloc(MAX_JPEG);
 if(!encrypted)abort();
 mbedtls_gcm_init(&gcm);
 if(mbedtls_gcm_setkey(&gcm,MBEDTLS_CIPHER_ID_AES,FRAME_KEY,256))abort();
 WiFi.mode(WIFI_AP);
 WiFi.softAPConfig(IPAddress(192,168,50,1),IPAddress(192,168,50,1),IPAddress(255,255,255,0));
 if(!WiFi.softAP(CAM_SSID,CAM_PASSWORD,6,false,1))abort();
 // The hardware RNG is only truly random once the radio is running, so the
 // boot nonce must be drawn after Wi-Fi starts or GCM nonces could repeat.
 esp_fill_random(bootNonce,sizeof bootNonce);
 server.begin();Serial.println("Camera ready: private AP, encrypted stream");
}
void loop(){
 if(!peer.connected()){
  peer.stop();peer=server.available();
  // Each new connection starts with the bridge's fresh challenge.
  if(peer&&!readExact(peer,challenge,CHALLENGE))peer.stop();
  // Every connection re-measures the light, so resetting the bridge re-locks exposure.
  if(peer){setAuto(true);lockAt=millis()+SETTLE_MS;locked=false;}
  delay(10);return;
 }
 if(!locked){
  // No frames are sent while exposure settles, so every received frame is a locked one.
  if(int32_t(millis()-lockAt)<0){delay(10);return;}
  setAuto(false);locked=true;
  // Drop the frames buffered before the lock.
  for(int i=0;i<2;++i){camera_fb_t* old=esp_camera_fb_get();if(old)esp_camera_fb_return(old);}
 }
 uint32_t start=millis();
 camera_fb_t* fb=esp_camera_fb_get();
 if(!fb){delay(500);return;}
 if(fb->len<=MAX_JPEG){
  uint8_t aad[HEADER+CHALLENGE]={ 'S','L','P','C',PROTOCOL_VERSION },tag[16];
  uint8_t* h=aad;
  put32(h+5,sequence);uint64_t t=esp_timer_get_time();
  for(int i=16;i>=9;--i){h[i]=t;t>>=8;}
  put32(h+17,fb->len);memcpy(h+21,bootNonce,8);put32(h+29,sequence);
  memcpy(aad+HEADER,challenge,CHALLENGE);
  // Authenticated data = header + this connection's challenge (challenge is not transmitted back).
  int err=mbedtls_gcm_crypt_and_tag(&gcm,MBEDTLS_GCM_ENCRYPT,fb->len,h+21,12,aad,sizeof aad,fb->buf,encrypted,16,tag);
  if(err||!writeExact(peer,h,HEADER)||!writeExact(peer,encrypted,fb->len)||!writeExact(peer,tag,16))peer.stop();
  if(++sequence==0)esp_restart();
 }
 esp_camera_fb_return(fb);
 // If sending takes longer than a frame period, the rate simply drops.
 uint32_t elapsed=millis()-start;if(elapsed<FRAME_MS)delay(FRAME_MS-elapsed);
}
