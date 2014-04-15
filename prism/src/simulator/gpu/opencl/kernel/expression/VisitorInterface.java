/**
 * 
 */
package simulator.gpu.opencl.kernel.expression;

/**
 * @author mcopik
 *
 */
public interface VisitorInterface
{
	void visit(Expression expr);

	void visit(Method method);

	void visit(ForLoop loop);
}
