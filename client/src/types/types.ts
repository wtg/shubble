class TextAnnotation extends window.mapkit.AnnotationView {
  constructor(annotation) {
    super(annotation);
    this.element.style.background = "none";
    this.element.style.border = "none";
    this.element.innerText = annotation.title;
  }
}