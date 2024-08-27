const text = '{"name":"John", "birthday":"1986-12-14"}';
const obj = JSON.parse(text);
obj.birthday = new Date(obj.birthday);