/////////////////////////////////////////////////////////////////////
////
//// AUTOGENERATED FILE - DO NOT EDIT
//// DRM SCRIPT VERSION 2.1.0
//// DRM HDK VERSION 4.1.0.0
//// DRM VERSION 4.1.0
//// COPYRIGHT (C) ALGODONE
////
/////////////////////////////////////////////////////////////////////

module drm_ip_activator_0x1003000e00010001_wrapper (
  // drm bus clock and reset
  input  wire         drm_aclk,
  input  wire         drm_arstn,
  // drm bus slave interface
  input  wire         drm_bus_slave_i_cs,
  input  wire         drm_bus_slave_i_cyc,
  input  wire         drm_bus_slave_i_we,
  input  wire [1:0]   drm_bus_slave_i_adr,
  input  wire         drm_bus_slave_i_dat,
  output wire         drm_bus_slave_o_ack,
  output wire         drm_bus_slave_o_sta,
  output wire         drm_bus_slave_o_intr,
  output wire         drm_bus_slave_o_dat,
  // ip core clock and reset
  input  wire         ip_core_aclk,
  input  wire         ip_core_arstn,
  // ip core interface
  input  wire         drm_event,
  input  wire         drm_arst,
  output wire         activation_code_ready,
  output wire         demo_mode,
  output wire [127:0] activation_code
);

  DRM_IP_ACTIVATOR_0x1003000E00010001 drm_ip_activator_0x1003000e00010001_inst (
    .DRM_ACLK              (drm_aclk),
    .DRM_ARSTN             (drm_arstn),
    .DRM_BUS_SLAVE_I_CS    (drm_bus_slave_i_cs),
    .DRM_BUS_SLAVE_I_CYC   (drm_bus_slave_i_cyc),
    .DRM_BUS_SLAVE_I_WE    (drm_bus_slave_i_we),
    .DRM_BUS_SLAVE_I_ADR   (drm_bus_slave_i_adr),
    .DRM_BUS_SLAVE_I_DAT   (drm_bus_slave_i_dat),
    .DRM_BUS_SLAVE_O_ACK   (drm_bus_slave_o_ack),
    .DRM_BUS_SLAVE_O_STA   (drm_bus_slave_o_sta),
    .DRM_BUS_SLAVE_O_INTR  (drm_bus_slave_o_intr),
    .DRM_BUS_SLAVE_O_DAT   (drm_bus_slave_o_dat),
    .IP_CORE_ACLK          (ip_core_aclk),
    .IP_CORE_ARSTN         (ip_core_arstn),
    .DRM_EVENT             (drm_event),
    .DRM_ARST              (drm_arst),
    .ACTIVATION_CODE_READY (activation_code_ready),
    .DEMO_MODE             (demo_mode),
    .ACTIVATION_CODE       (activation_code)
  );
    
endmodule