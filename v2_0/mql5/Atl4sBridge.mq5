//+------------------------------------------------------------------+
//|                                                  Atl4sBridge.mq5 |
//|                                                     Atl4s-Forex  |
//|                                       https://www.atl4s-forex.com|
//+------------------------------------------------------------------+
#property copyright "Atl4s-Forex"
#property link      "https://www.atl4s-forex.com"
#property version   "2.00"
#include <Trade\Trade.mqh>

CTrade trade;

// Native Socket Functions require "Allow WebRequest" for localhost
// Tools -> Options -> Expert Advisors -> Allow WebRequest for listed URL:
// Add "http://localhost" and "http://127.0.0.1" just in case (though sockets are different, sometimes MT5 checks this)

input int InpPort = 5557; // Port to connect to Python

int socket_handle = INVALID_HANDLE;
string host = "127.0.0.1";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   Print("Atl4sBridge: Initializing Native Socket Client...");
   
   // Create Socket
   socket_handle = SocketCreate();
   if(socket_handle == INVALID_HANDLE) {
      Print("Error creating socket: ", GetLastError());
      return(INIT_FAILED);
   }
   
   // Connect to Python Server
   if(!SocketConnect(socket_handle, host, InpPort, 1000)) {
      Print("Error connecting to Python Server: ", GetLastError());
      Print("Make sure main.py is running FIRST!");
      return(INIT_FAILED);
   }
   
   Print("Atl4sBridge: Connected to Python Server.");
   EventSetMillisecondTimer(100); // Check for commands
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   SocketClose(socket_handle);
   Print("Atl4sBridge: Deinitialized.");
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(socket_handle == INVALID_HANDLE) return;
   
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;
   
   // Format: TICK|SYMBOL|TIME|BID|ASK|LAST|VOL\n
   string msg = StringFormat("TICK|%s|%I64d|%.5f|%.5f|%.5f|%I64d\n", 
                             _Symbol, tick.time_msc, tick.bid, tick.ask, tick.last, tick.volume);
   
   // Send Tick
   uchar req[];
   StringToCharArray(msg, req);
   // Remove null terminator from array for sending
   int len = StringLen(msg);
   
   if(SocketSend(socket_handle, req, len) < 0) {
      int err = GetLastError();
      Print("Error sending tick: ", err);
      
      // Simple Reconnect Logic
      if(err == 5273 || err == 5272 || err == 5270) { // Connection lost/refused
         Print("Attempting to reconnect...");
         SocketClose(socket_handle);
         socket_handle = SocketCreate();
         if(socket_handle != INVALID_HANDLE) {
            if(SocketConnect(socket_handle, host, InpPort, 1000)) {
               Print("Reconnected to Python Server.");
            }
         }
         }

      }

   
   // Check for commands on Tick as well (faster response)
   ReadCommands();
  }
//+------------------------------------------------------------------+
//| Process Incoming Commands                                        |
//+------------------------------------------------------------------+
void ProcessCommand(string cmd)
  {
   string parts[];
   ushort sep = StringGetCharacter("|", 0);
   int count = StringSplit(cmd, sep, parts);
   
   if(count > 0)
     {
      if(parts[0] == "OPEN_TRADE")
        {
         if(count < 6)
           {
            Print("Error: Malformed OPEN_TRADE command. Expected 6 parts, got ", count);
            return;
           }
           
         string symbol = parts[1];
         int type = (int)StringToInteger(parts[2]);
         double volume = StringToDouble(parts[3]);
         double sl = StringToDouble(parts[4]);
         double tp = StringToDouble(parts[5]);
         
         Print("Processing Trade: ", (type==0 ? "BUY" : "SELL"), " ", volume, " lots on ", symbol);
         
         if(type == 0) // BUY
           {
            if(trade.Buy(volume, symbol, 0, sl, tp, "Atl4s-Bot"))
               Print("Buy Order Executed Successfully. Ticket: ", trade.ResultOrder());
            else
              {
               int err = trade.ResultRetcode();
               Print("Buy Order Failed. Error: ", err, " - ", trade.ResultRetcodeDescription());
               
               // Retry for Market Execution (Error 10016)
               if(err == 10016)
                 {
                  Print("Retrying with Market Execution (0 SL/TP)...");
                  if(trade.Buy(volume, symbol, 0, 0, 0, "Atl4s-Bot"))
                    {
                     ulong ticket = trade.ResultOrder();
                     Print("Market Entry Success. Ticket: ", ticket, ". Applying Stops...");
                     Sleep(200); // Small wait for position registration
                     if(trade.PositionModify(ticket, sl, tp))
                        Print("Stops Applied Successfully.");
                     else
                        Print("Failed to apply stops: ", trade.ResultRetcode());
                    }
                 }
              }
           }
         else if(type == 1) // SELL
           {
            if(trade.Sell(volume, symbol, 0, sl, tp, "Atl4s-Bot"))
               Print("Sell Order Executed Successfully. Ticket: ", trade.ResultOrder());
            else
              {
               int err = trade.ResultRetcode();
               Print("Sell Order Failed. Error: ", err, " - ", trade.ResultRetcodeDescription());
               
               // Retry for Market Execution (Error 10016)
               if(err == 10016)
                 {
                  Print("Retrying with Market Execution (0 SL/TP)...");
                  if(trade.Sell(volume, symbol, 0, 0, 0, "Atl4s-Bot"))
                    {
                     ulong ticket = trade.ResultOrder();
                     Print("Market Entry Success. Ticket: ", ticket, ". Applying Stops...");
                     Sleep(200); // Small wait for position registration
                     if(trade.PositionModify(ticket, sl, tp))
                        Print("Stops Applied Successfully.");
                     else
                        Print("Failed to apply stops: ", trade.ResultRetcode());
                    }
                 }
              }
           }
        }
         else if(parts[0] == "MODIFY_TRADE")
           {
            // MODIFY_TRADE|TICKET|SL|TP
            if(count < 4) { Print("Error: Malformed MODIFY_TRADE"); return; }
            ulong ticket = (ulong)StringToInteger(parts[1]);
            double sl = StringToDouble(parts[2]);
            double tp = StringToDouble(parts[3]);
            
            if(trade.PositionModify(ticket, sl, tp))
               Print("Modify Trade Success: Ticket ", ticket);
            else
               Print("Modify Trade Failed: ", trade.ResultRetcode());
           }
         else if(parts[0] == "CLOSE_TRADE")
           {
            // CLOSE_TRADE|TICKET
            if(count < 2) { Print("Error: Malformed CLOSE_TRADE"); return; }
            ulong ticket = (ulong)StringToInteger(parts[1]);
            
            if(trade.PositionClose(ticket))
               Print("Close Trade Success: Ticket ", ticket);
            else
               Print("Close Trade Failed: ", trade.ResultRetcode());
           }
         else if(parts[0] == "CLOSE_PARTIAL")
           {
            // CLOSE_PARTIAL|TICKET|VOLUME
            if(count < 3) { Print("Error: Malformed CLOSE_PARTIAL"); return; }
            ulong ticket = (ulong)StringToInteger(parts[1]);
            double volume = StringToDouble(parts[2]);
            
            if(trade.PositionClosePartial(ticket, volume))
               Print("Partial Close Success: Ticket ", ticket, " Vol: ", volume);
            else
               Print("Partial Close Failed: ", trade.ResultRetcode());
           }
     }
  }

//+------------------------------------------------------------------+
//| Read Commands from Socket                                        |
//+------------------------------------------------------------------+
void ReadCommands()
  {
   if(socket_handle == INVALID_HANDLE) return;
   
   uint readable = SocketIsReadable(socket_handle);
   if(readable > 0)
     {
      Print("Debug: Socket has ", readable, " bytes readable.");
      
      uchar rsp[];
      ArrayResize(rsp, readable);
      int len = SocketRead(socket_handle, rsp, readable, 1000);
      
      if(len > 0)
        {
         string cmd = CharArrayToString(rsp, 0, len);
         Print("Received Command: ", cmd);
         
         string commands[];
         ushort sep = StringGetCharacter("\n", 0);
         int cmd_count = StringSplit(cmd, sep, commands);
         
         for(int i=0; i<cmd_count; i++) {
            string c = commands[i];
            StringTrimRight(c);
            StringTrimLeft(c);
            if(StringLen(c) > 0) {
               ProcessCommand(c);
            }
         }
        }
      else
        {
         Print("Error: SocketRead failed or returned 0. Error: ", GetLastError());
        }
     }
  }

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
   ReadCommands();
  }
//+------------------------------------------------------------------+
