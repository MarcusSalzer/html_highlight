Future<void> play(String filePath) async {
  if (_player.state == PlayerState.playing) {
    print("playing, stops");
    _player.stop();
  }
  assert(_player.state == PlayerState.stopped ||
      _player.state == PlayerState.completed);

  await _player.setSourceDeviceFile(filePath);
}