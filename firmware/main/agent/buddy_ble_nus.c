/*
 * buddy_ble_nus — NimBLE Nordic UART Service peripheral. See buddy_ble_nus.h.
 *
 * Modeled on the factory bleprph scaffold (main/hal/utils/bleprph), trimmed
 * to a single NUS service and wired for LE Secure Connections bonding with a
 * DisplayOnly passkey (shown on the avatar screen by the C++ layer).
 */
#include "buddy_ble_nus.h"

#include <string.h>

#include "esp_log.h"
#include "esp_mac.h"
#include "esp_random.h"

#include "host/ble_hs.h"
#include "host/ble_uuid.h"
#include "host/util/util.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

/* Provided by the NimBLE store_config component (same as the scaffold). */
void ble_store_config_init(void);
/* Erase all bonds. Declared in store/ble_store.h but pull it in directly. */
int ble_store_clear(void);

static const char *TAG = "buddy.ble";

/* Nordic UART Service UUIDs (128-bit, little-endian byte order for NimBLE).
 *   Service: 6e400001-b5a3-f393-e0a9-e50e24dcca9e
 *   RX (write, desktop->device): 6e400002-...
 *   TX (notify, device->desktop): 6e400003-...
 */
static const ble_uuid128_t NUS_SVC_UUID = BLE_UUID128_INIT(
    0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0,
    0x93, 0xf3, 0xa3, 0xb5, 0x01, 0x00, 0x40, 0x6e);
static const ble_uuid128_t NUS_RX_UUID = BLE_UUID128_INIT(
    0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0,
    0x93, 0xf3, 0xa3, 0xb5, 0x02, 0x00, 0x40, 0x6e);
static const ble_uuid128_t NUS_TX_UUID = BLE_UUID128_INIT(
    0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0,
    0x93, 0xf3, 0xa3, 0xb5, 0x03, 0x00, 0x40, 0x6e);

#define BUDDY_DEVICE_NAME "Claude StackChan"
#define BUDDY_RX_ACC_MAX  2048

static buddy_line_cb_t s_on_line = NULL;
static uint8_t s_own_addr_type;
static uint16_t s_conn_handle = BLE_HS_CONN_HANDLE_NONE;
static uint16_t s_tx_handle = 0;     /* value handle of the TX characteristic */
static bool s_authenticated = false;

/* RX reassembly buffer — accumulate write fragments until we hit '\n'. */
static char s_rx_acc[BUDDY_RX_ACC_MAX + 1];
static int s_rx_len = 0;

static void buddy_advertise(void);

/* ---- RX: write-callback feeds the newline splitter ---- */

static void rx_feed(const uint8_t *data, int len)
{
    for (int i = 0; i < len; i++) {
        char c = (char)data[i];
        if (c == '\n') {
            if (s_rx_len > 0 && s_on_line) {
                s_rx_acc[s_rx_len] = '\0';
                s_on_line(s_rx_acc, s_rx_len);
            }
            s_rx_len = 0;
            continue;
        }
        if (c == '\r') continue;
        if (s_rx_len >= BUDDY_RX_ACC_MAX) {
            /* Overlong line (no newline yet) — drop it to resync. */
            ESP_LOGW(TAG, "RX line overflow (%d bytes), dropping", s_rx_len);
            s_rx_len = 0;
        }
        s_rx_acc[s_rx_len++] = c;
    }
}

static int nus_access_cb(uint16_t conn_handle, uint16_t attr_handle,
                         struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    if (ctxt->op == BLE_GATT_ACCESS_OP_WRITE_CHR) {
        /* The characteristic flags already demand an authenticated link; check
         * it here too, because everything this feeds — settimeofday, the bond
         * wipe, arbitrary on-screen "approve:" text — is reachable from a
         * single unchecked write. */
        struct ble_gap_conn_desc desc;
        if (ble_gap_conn_find(conn_handle, &desc) != 0 ||
            !desc.sec_state.encrypted || !desc.sec_state.authenticated) {
            return BLE_ATT_ERR_INSUFFICIENT_AUTHEN;
        }
        /* One ATT write value is bounded by the ATT MTU (512), so a 512-byte
         * scratch copy holds it whole. JSON lines longer than one write are
         * reassembled across writes by rx_feed's newline accumulation. */
        uint8_t buf[512];
        uint16_t out_len = 0;
        int rc = ble_hs_mbuf_to_flat(ctxt->om, buf, sizeof(buf), &out_len);
        if (rc != 0) return BLE_ATT_ERR_UNLIKELY;
        rx_feed(buf, out_len);
        return 0;
    }
    return BLE_ATT_ERR_UNLIKELY;
}

static const struct ble_gatt_svc_def s_gatt_svcs[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &NUS_SVC_UUID.u,
        .characteristics = (struct ble_gatt_chr_def[]){
            {
                /* RX: desktop writes JSON lines here. WRITE_ENC alone is
                 * satisfied by *any* encrypted link, and a central that
                 * declares NoInputNoOutput negotiates Just Works — no passkey
                 * is ever displayed and sm_mitm only sets a bit in the AuthReq.
                 * WRITE_AUTHEN is what actually requires the MITM-protected
                 * pairing this device shows a PIN for. */
                .uuid = &NUS_RX_UUID.u,
                .access_cb = nus_access_cb,
                .flags = BLE_GATT_CHR_F_WRITE | BLE_GATT_CHR_F_WRITE_NO_RSP |
                         BLE_GATT_CHR_F_WRITE_ENC | BLE_GATT_CHR_F_WRITE_AUTHEN,
            },
            {
                /* TX: device notifies JSON lines back. Notify-only; the link
                 * is already authenticated because RX write requires it. */
                .uuid = &NUS_TX_UUID.u,
                .access_cb = nus_access_cb,
                .flags = BLE_GATT_CHR_F_NOTIFY,
                .val_handle = &s_tx_handle,
            },
            {0},
        },
    },
    {0},
};

/* ---- TX: notify a line back to the desktop ---- */

int buddy_nus_notify(const char *data, int len)
{
    if (s_conn_handle == BLE_HS_CONN_HANDLE_NONE || s_tx_handle == 0) {
        return -1;
    }
    struct os_mbuf *om = ble_hs_mbuf_from_flat(data, len);
    if (!om) return -2;
    return ble_gatts_notify_custom(s_conn_handle, s_tx_handle, om);
}

bool buddy_nus_connected(void)
{
    return s_conn_handle != BLE_HS_CONN_HANDLE_NONE;
}

bool buddy_nus_authenticated(void)
{
    return s_conn_handle != BLE_HS_CONN_HANDLE_NONE && s_authenticated;
}

void buddy_nus_ensure_advertising(void)
{
    /* buddy_advertise() is only ever re-armed from GAP edges (sync, disconnect,
     * failed connect) and each of its failure paths just logs and returns, so
     * one unlucky ble_gap_adv_start() leaves this always-on device silently
     * undiscoverable until a NimBLE host reset or a power cycle. Nothing else
     * polls ble_gap_adv_active(). */
    if (!ble_hs_is_enabled() || !ble_hs_synced()) return;
    if (s_conn_handle != BLE_HS_CONN_HANDLE_NONE) return;
    if (ble_gap_adv_active()) return;
    buddy_advertise();
}

void buddy_nus_unpair(void)
{
    int rc = ble_store_clear();
    ESP_LOGI(TAG, "unpair: ble_store_clear rc=%d", rc);
    /* Drop the live link too so the central re-pairs cleanly. */
    if (s_conn_handle != BLE_HS_CONN_HANDLE_NONE) {
        ble_gap_terminate(s_conn_handle, BLE_ERR_REM_USER_CONN_TERM);
    }
}

/* ---- GAP ---- */

static int buddy_gap_event(struct ble_gap_event *event, void *arg)
{
    struct ble_gap_conn_desc desc;

    switch (event->type) {
        case BLE_GAP_EVENT_CONNECT:
            if (event->connect.status == 0) {
                s_conn_handle = event->connect.conn_handle;
                s_authenticated = false;
                s_rx_len = 0;
                ESP_LOGI(TAG, "central connected; handle=%d", s_conn_handle);
                /* Ask the central to encrypt straight away (triggers SC
                 * pairing on first connect, encryption on a bonded one). */
                ble_gap_security_initiate(s_conn_handle);
                buddy_on_connect();
            } else {
                ESP_LOGW(TAG, "connect failed; status=%d", event->connect.status);
                buddy_advertise();
            }
            return 0;

        case BLE_GAP_EVENT_DISCONNECT:
            ESP_LOGI(TAG, "central disconnected; reason=%d", event->disconnect.reason);
            s_conn_handle = BLE_HS_CONN_HANDLE_NONE;
            s_authenticated = false;
            s_rx_len = 0;
            buddy_on_disconnect();
            buddy_advertise();
            return 0;

        case BLE_GAP_EVENT_CONN_UPDATE:
        case BLE_GAP_EVENT_MTU:
            return 0;

        case BLE_GAP_EVENT_ADV_COMPLETE:
            buddy_advertise();
            return 0;

        case BLE_GAP_EVENT_ENC_CHANGE:
            if (ble_gap_conn_find(event->enc_change.conn_handle, &desc) == 0) {
                s_authenticated = desc.sec_state.encrypted && desc.sec_state.authenticated;
                ESP_LOGI(TAG, "enc change; status=%d encrypted=%d authenticated=%d bonded=%d",
                         event->enc_change.status, desc.sec_state.encrypted,
                         desc.sec_state.authenticated, desc.sec_state.bonded);
            }
            /* NimBLE leaves the ACL up when pairing fails or the 30 s SM timer
             * expires, so this status is the only notice the UI ever gets that
             * the PIN on screen has become meaningless. */
            if (event->enc_change.status != 0) {
                buddy_on_pairing_failed();
            }
            return 0;

        case BLE_GAP_EVENT_REPEAT_PAIRING:
            /* Peer wants to re-pair though we have a bond: drop the old bond
             * and let pairing proceed (matches the scaffold's policy). */
            if (ble_gap_conn_find(event->repeat_pairing.conn_handle, &desc) == 0) {
                ble_store_util_delete_peer(&desc.peer_id_addr);
            }
            return BLE_GAP_REPEAT_PAIRING_RETRY;

        case BLE_GAP_EVENT_PASSKEY_ACTION: {
            struct ble_sm_io pkey = {0};
            if (event->passkey.params.action == BLE_SM_IOACT_DISP) {
                uint32_t passkey = esp_random() % 1000000u;
                pkey.action = BLE_SM_IOACT_DISP;
                pkey.passkey = passkey;
                int rc = ble_sm_inject_io(event->passkey.conn_handle, &pkey);
                ESP_LOGI(TAG, "display passkey %06" PRIu32 " (inject rc=%d)", passkey, rc);
                buddy_on_passkey(passkey);
            } else if (event->passkey.params.action == BLE_SM_IOACT_NUMCMP) {
                pkey.action = BLE_SM_IOACT_NUMCMP;
                pkey.numcmp_accept = 1;
                ble_sm_inject_io(event->passkey.conn_handle, &pkey);
            }
            return 0;
        }

        case BLE_GAP_EVENT_SUBSCRIBE:
            ESP_LOGI(TAG, "subscribe; attr=%d cur_notify=%d",
                     event->subscribe.attr_handle, event->subscribe.cur_notify);
            return 0;

        default:
            return 0;
    }
}

/* ---- Advertising ---- */

static void buddy_advertise(void)
{
    if (ble_gap_adv_active()) return;

    struct ble_hs_adv_fields fields;
    struct ble_hs_adv_fields rsp_fields;
    int rc;

    /* Adv packet: flags + 128-bit NUS service UUID (doesn't fit alongside the
     * full name, so the name goes in the scan response). */
    memset(&fields, 0, sizeof fields);
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.uuids128 = (ble_uuid128_t *)&NUS_SVC_UUID;
    fields.num_uuids128 = 1;
    fields.uuids128_is_complete = 1;
    rc = ble_gap_adv_set_fields(&fields);
    if (rc != 0) {
        ESP_LOGE(TAG, "adv_set_fields rc=%d", rc);
        return;
    }

    /* Scan response: complete device name + TX power. */
    memset(&rsp_fields, 0, sizeof rsp_fields);
    const char *name = ble_svc_gap_device_name();
    rsp_fields.name = (uint8_t *)name;
    rsp_fields.name_len = strlen(name);
    rsp_fields.name_is_complete = 1;
    rsp_fields.tx_pwr_lvl_is_present = 1;
    rsp_fields.tx_pwr_lvl = BLE_HS_ADV_TX_PWR_LVL_AUTO;
    rc = ble_gap_adv_rsp_set_fields(&rsp_fields);
    if (rc != 0) {
        ESP_LOGE(TAG, "adv_rsp_set_fields rc=%d", rc);
        return;
    }

    struct ble_gap_adv_params adv_params;
    memset(&adv_params, 0, sizeof adv_params);
    adv_params.conn_mode = BLE_GAP_CONN_MODE_UND;
    adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;
    rc = ble_gap_adv_start(s_own_addr_type, NULL, BLE_HS_FOREVER, &adv_params,
                           buddy_gap_event, NULL);
    if (rc != 0) {
        ESP_LOGE(TAG, "adv_start rc=%d", rc);
        return;
    }
    ESP_LOGI(TAG, "advertising as \"%s\"", name);
}

static void buddy_on_sync(void)
{
    int rc = ble_hs_util_ensure_addr(0);
    if (rc != 0) {
        ESP_LOGE(TAG, "ensure_addr rc=%d", rc);
        return;
    }
    rc = ble_hs_id_infer_auto(0, &s_own_addr_type);
    if (rc != 0) {
        ESP_LOGE(TAG, "infer_auto rc=%d", rc);
        return;
    }
    buddy_advertise();
}

static void buddy_on_reset(int reason)
{
    ESP_LOGW(TAG, "nimble reset; reason=%d", reason);
}

static void buddy_host_task(void *param)
{
    nimble_port_run();
    nimble_port_freertos_deinit();
}

void buddy_nus_init(buddy_line_cb_t on_line)
{
    s_on_line = on_line;

    esp_err_t ret = nimble_port_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "nimble_port_init failed: %d", ret);
        return;
    }

    ble_hs_cfg.reset_cb = buddy_on_reset;
    ble_hs_cfg.sync_cb = buddy_on_sync;
    ble_hs_cfg.gatts_register_cb = NULL;
    ble_hs_cfg.store_status_cb = ble_store_util_status_rr;

    /* LE Secure Connections bonding, DisplayOnly (we show a passkey on the
     * avatar screen for the user to type into the desktop app). */
    ble_hs_cfg.sm_io_cap = BLE_HS_IO_DISPLAY_ONLY;
    ble_hs_cfg.sm_bonding = 1;
    ble_hs_cfg.sm_mitm = 1;
    ble_hs_cfg.sm_sc = 1;
    ble_hs_cfg.sm_our_key_dist = BLE_SM_PAIR_KEY_DIST_ENC | BLE_SM_PAIR_KEY_DIST_ID;
    ble_hs_cfg.sm_their_key_dist = BLE_SM_PAIR_KEY_DIST_ENC | BLE_SM_PAIR_KEY_DIST_ID;

    int rc = ble_gatts_count_cfg(s_gatt_svcs);
    if (rc != 0) {
        ESP_LOGE(TAG, "gatts_count_cfg rc=%d", rc);
        return;
    }
    rc = ble_gatts_add_svcs(s_gatt_svcs);
    if (rc != 0) {
        ESP_LOGE(TAG, "gatts_add_svcs rc=%d", rc);
        return;
    }

    rc = ble_svc_gap_device_name_set(BUDDY_DEVICE_NAME);
    if (rc != 0) ESP_LOGW(TAG, "device_name_set rc=%d", rc);

    ble_store_config_init();
    nimble_port_freertos_init(buddy_host_task);
    ESP_LOGI(TAG, "NimBLE NUS peripheral started");
}
