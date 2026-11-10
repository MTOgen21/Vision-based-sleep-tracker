#pragma once
#include <Arduino.h>
#include <WiFi.h>
#include <esp_system.h>
#include <mbedtls/gcm.h>
#include "secrets.h"
constexpr uint32_t MAX_JPEG = 64 * 1024;
constexpr size_t HEADER = 33;
constexpr uint8_t PROTOCOL_VERSION = 2;
// Fresh random value sent by the bridge on every TCP connection. The camera
// authenticates it with every frame, so frames from earlier connections fail.
constexpr size_t CHALLENGE = 16;
// Must match save_jpegs.py --baud (default 921600).
constexpr uint32_t USB_BAUD = 921600;
inline void put32(uint8_t* p,uint32_t v){for(int i=3;i>=0;--i){p[i]=v;v>>=8;}}
inline uint32_t get32(const uint8_t* p){return (uint32_t(p[0])<<24)|(uint32_t(p[1])<<16)|(uint32_t(p[2])<<8)|p[3];}
inline bool readExact(WiFiClient& c,uint8_t* p,size_t n){
  uint32_t last=millis();
  while(n){int r=c.read(p,n);if(r>0){p+=r;n-=r;last=millis();}else{if(!c.connected()||millis()-last>5000)return false;delay(1);}}
  return true;
}
inline bool writeExact(WiFiClient& c,const uint8_t* p,size_t n){
  while(n){size_t r=c.write(p,n);if(!r)return false;p+=r;n-=r;}return true;
}
