// Preview selection only. These coordinates must be mapped to camera-frame
// coordinates (including aspect-fill cropping) before the counting core uses them.
Page({
  data: {
    cameraEnabled: false,
    cameraReady: false,
    cameraError: '',
    roiSelected: false,
    selectionStyle: '',
    selectionMessage: '',
    previewRegion: null
  },

  toggleCamera() {
    if (this.data.cameraEnabled) {
      this.stopCamera();
      return;
    }
    this.setData({ cameraEnabled: true, cameraReady: false, cameraError: '' });
  },

  onCameraReady() {
    if (!this.data.cameraEnabled) return;
    this.setData({ cameraReady: true });
    this.measurePreview();
  },

  measurePreview() {
    this._previewBounds = null;
    wx.createSelectorQuery().in(this).select('.preview').boundingClientRect(rect => {
      if (this.data.cameraEnabled && rect) this._previewBounds = rect;
    }).exec();
  },

  stopCamera() {
    this._dragStart = null;
    this._previewBounds = null;
    this.setData({ cameraEnabled: false, cameraReady: false });
    this.clearSelection();
  },

  onCameraError(event) {
    this.stopCamera();
    this.setData({ cameraError: (event.detail && event.detail.errMsg) || '相机无法开启，请检查相机权限后重试。' });
  },

  onCameraStop() {
    if (!this.data.cameraEnabled) return;
    this.stopCamera();
    this.setData({ cameraError: '相机已中断，请重新开启并圈选光屏。' });
  },

  clearSelection() {
    this._dragStart = null;
    this.setData({ roiSelected: false, selectionStyle: '', previewRegion: null, selectionMessage: '' });
  },

  previewPoint(touch) {
    const rect = this._previewBounds;
    if (!rect || !touch || !Number.isFinite(touch.clientX) || !Number.isFinite(touch.clientY)) return null;
    return {
      x: Math.max(0, Math.min(rect.width, touch.clientX - rect.left)),
      y: Math.max(0, Math.min(rect.height, touch.clientY - rect.top))
    };
  },

  onSelectionStart(event) {
    if (!this.data.cameraReady) return;
    this._dragStart = this.previewPoint(event.touches[0]);
    if (this._dragStart) this.setData({ roiSelected: false, previewRegion: null, selectionStyle: '', selectionMessage: '' });
  },

  onSelectionMove(event) {
    if (!this._dragStart) return;
    const region = this.regionFromTouch(event.touches[0]);
    if (region) this.drawRegion(region);
  },

  regionFromTouch(touch) {
    const point = this.previewPoint(touch);
    if (!point || !this._dragStart) return null;
    return {
      x: Math.min(point.x, this._dragStart.x),
      y: Math.min(point.y, this._dragStart.y),
      width: Math.abs(point.x - this._dragStart.x),
      height: Math.abs(point.y - this._dragStart.y)
    };
  },

  drawRegion(region) {
    this.setData({ selectionStyle: `left:${region.x}px;top:${region.y}px;width:${region.width}px;height:${region.height}px;` });
  },

  onSelectionEnd(event) {
    const region = this.regionFromTouch(event.changedTouches[0]);
    this._dragStart = null;
    if (!region) return;
    if (region.width < 32 || region.height < 32) {
      this.clearSelection();
      this.setData({ selectionMessage: '选择区域太小，请框住整个光屏。' });
      return;
    }
    const bounds = this._previewBounds;
    this.drawRegion(region);
    this.setData({
      roiSelected: true,
      previewRegion: { x: region.x / bounds.width, y: region.y / bounds.height, width: region.width / bounds.width, height: region.height / bounds.height },
      selectionMessage: '已圈选光屏。当前版本尚未开始测量。'
    });
  },

  onSelectionCancel() { this.clearSelection(); },
  onResize() {
    this.clearSelection();
    if (this.data.cameraEnabled) this.measurePreview();
  },
  onHide() { this.stopCamera(); },
  onUnload() { this.stopCamera(); }
});
